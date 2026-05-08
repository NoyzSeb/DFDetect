from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import make_grid

try:
	import wandb
except Exception:  # pragma: no cover - optional dependency at runtime
	wandb = None


def get_next_run_name(prefix: str = "EfNetwUNet", counter_file: str = "./runs/run_counter.txt") -> Tuple[str, Path]:
	"""Return a new run name and its checkpoint directory.

	Counter persists in counter_file and increments per call.
	"""
	counter_path = Path(counter_file)
	counter_path.parent.mkdir(parents=True, exist_ok=True)

	if counter_path.exists():
		raw = counter_path.read_text(encoding="utf-8").strip()
		current = int(raw) if raw.isdigit() else 0
	else:
		current = 0

	next_val = current + 1
	counter_path.write_text(str(next_val), encoding="utf-8")

	run_name = f"{prefix}_{next_val:02d}"
	checkpoint_dir = counter_path.parent / run_name
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	return run_name, checkpoint_dir


def compute_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
	"""Compute Dice score per-sample, expects N,C,H,W or C,H,W."""
	if pred.shape != target.shape:
		raise ValueError("pred and target must have the same shape.")
	if pred.dim() == 3:
		pred = pred.unsqueeze(0)
		target = target.unsqueeze(0)
	elif pred.dim() < 3:
		raise ValueError("Tensors must have shape (C, H, W) or (N, C, H, W).")

	pred_flat = pred.flatten(1)
	target_flat = target.flatten(1)
	intersection = (pred_flat * target_flat).sum(dim=1)
	dice = (2 * intersection + eps) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + eps)
	return dice


def compute_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
	"""Compute IoU per-sample, expects N,C,H,W or C,H,W."""
	if pred.shape != target.shape:
		raise ValueError("pred and target must have the same shape.")
	if pred.dim() == 3:
		pred = pred.unsqueeze(0)
		target = target.unsqueeze(0)
	elif pred.dim() < 3:
		raise ValueError("Tensors must have shape (C, H, W) or (N, C, H, W).")

	pred_flat = pred.flatten(1)
	target_flat = target.flatten(1)
	intersection = (pred_flat * target_flat).sum(dim=1)
	union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
	return (intersection + eps) / (union + eps)


def compute_cosine_sim(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
	"""Compute cosine similarity per-sample, expects N,C,H,W or C,H,W."""
	if pred.shape != target.shape:
		raise ValueError("pred and target must have the same shape.")
	if pred.dim() == 3:
		pred = pred.unsqueeze(0)
		target = target.unsqueeze(0)
	elif pred.dim() < 3:
		raise ValueError("Tensors must have shape (C, H, W) or (N, C, H, W).")

	pred_flat = pred.flatten(1)
	target_flat = target.flatten(1)
	return F.cosine_similarity(pred_flat, target_flat, dim=1, eps=eps)


def normalize_fft_for_viewing(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
	"""Log-scale and min-max normalize for visualization."""
	if tensor.dim() < 3:
		raise ValueError("Tensor must have shape (C, H, W) or (N, C, H, W).")

	log_scaled = torch.log1p(torch.abs(tensor))
	if log_scaled.dim() == 3:
		log_scaled = log_scaled.unsqueeze(0)
	view = log_scaled.clone()
	view = view.flatten(2)
	min_val = view.min(dim=2, keepdim=True).values
	max_val = view.max(dim=2, keepdim=True).values
	view = (view - min_val) / (max_val - min_val + eps)
	return view.view_as(log_scaled)


class WandbLogger:
	"""Weights & Biases logger for dual-stream EfficientNet + U-Net training."""

	def __init__(self, project_name: str, config: Dict, run_name: Optional[str] = None) -> None:
		self.enabled = wandb is not None
		self.project_name = project_name
		self.config = config

		if run_name is None:
			run_name, checkpoint_dir = get_next_run_name()
		else:
			checkpoint_dir = Path("./runs") / run_name
			checkpoint_dir.mkdir(parents=True, exist_ok=True)

		self.run_name = run_name
		self.checkpoint_dir = checkpoint_dir

		self._sample_indices: Optional[torch.Tensor] = None
		self._visual_samples = int(config.get("visual_samples", 6))

		if self.enabled:
			mode = os.environ.get("WANDB_MODE")
			self.run = wandb.init(project=project_name, name=run_name, config=config, mode=mode)
		else:
			self.run = None

	def _set_sample_indices(self, batch_size: int) -> None:
		count = min(self._visual_samples, batch_size)
		self._sample_indices = torch.arange(count)

	@staticmethod
	def _to_3ch(tensor: torch.Tensor) -> torch.Tensor:
		if tensor.dim() == 3:
			tensor = tensor.unsqueeze(0)
		if tensor.size(1) == 1:
			return tensor.repeat(1, 3, 1, 1)
		return tensor

	@staticmethod
	def _normalize_01(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
		view = tensor.flatten(2)
		min_val = view.min(dim=2, keepdim=True).values
		max_val = view.max(dim=2, keepdim=True).values
		return ((view - min_val) / (max_val - min_val + eps)).view_as(tensor)

	def _compose_grid(self, tensors: Iterable[torch.Tensor]) -> torch.Tensor:
		tensor_list = list(tensors)
		composites = []
		for idx in range(tensor_list[0].size(0)):
			parts = [tensor[idx] for tensor in tensor_list]
			composite = torch.cat(parts, dim=2)
			composites.append(composite)

		grid = make_grid(torch.stack(composites, dim=0), nrow=1)
		return grid

	def log_train_step(self, losses_dict: Dict[str, float], lr: float, grad_norm: float, step: int) -> None:
		if not self.enabled:
			return

		payload = {
			"train/lr": lr,
			"train/grad_norm": grad_norm,
			"train/loss_total": losses_dict.get("loss_total"),
			"train/loss_spatial_cos": losses_dict.get("loss_spatial_cos"),
			"train/loss_spatial_mag": losses_dict.get("loss_spatial_mag"),
			"train/loss_freq_cos": losses_dict.get("loss_freq_cos"),
			"train/loss_freq_mag": losses_dict.get("loss_freq_mag"),
			"train/loss_consistency": losses_dict.get("loss_consistency"),
			"train/loss_cls": losses_dict.get("loss_cls"),
		}
		self.run.log(payload, step=step)

	def log_epoch_metrics(
		self,
		metrics_dict: Dict[str, float],
		epoch: int,
		split: str,
		*,
		step: int | None = None,
	) -> None:
		if not self.enabled:
			return

		payload = {
			f"{split}/{k}": v
			for k, v in metrics_dict.items()
			if k not in {"y_true", "y_pred", "class_names"}
		}
		self.run.log(payload, step=epoch if step is None else step)

		if "y_true" in metrics_dict and "y_pred" in metrics_dict:
			class_names = metrics_dict.get("class_names", ["real", "fake"])
			cm = wandb.plot.confusion_matrix(
				y_true=metrics_dict["y_true"],
				preds=metrics_dict["y_pred"],
				class_names=class_names,
			)
			self.run.log({f"{split}/confusion_matrix": cm}, step=epoch if step is None else step)

	def log_visuals(
		self,
		input_img: torch.Tensor,
		gt_spatial: torch.Tensor,
		gt_freq: torch.Tensor,
		pred_spatial: torch.Tensor,
		pred_freq: torch.Tensor,
		epoch: int,
		split: str,
		*,
		step: int | None = None,
	) -> None:
		if not self.enabled:
			return

		if epoch < 10:
			every = 1
		else:
			every = 5
		if epoch % every != 0:
			return

		batch_size = input_img.size(0)
		if self._sample_indices is None:
			self._set_sample_indices(batch_size)

		indices = self._sample_indices
		input_img = input_img[indices]
		gt_spatial = gt_spatial[indices]
		gt_freq = gt_freq[indices]
		pred_spatial = pred_spatial[indices]
		pred_freq = pred_freq[indices]

		input_view = self._normalize_01(self._to_3ch(input_img))
		gt_spatial_view = self._normalize_01(self._to_3ch(gt_spatial))
		pred_spatial_view = self._normalize_01(self._to_3ch(pred_spatial))
		gt_freq_view = self._normalize_01(self._to_3ch(normalize_fft_for_viewing(gt_freq)))
		pred_freq_view = self._normalize_01(self._to_3ch(normalize_fft_for_viewing(pred_freq)))

		spatial_grid = self._compose_grid([input_view, gt_spatial_view, pred_spatial_view])
		freq_grid = self._compose_grid([input_view, gt_freq_view, pred_freq_view])
		full_grid = self._compose_grid(
			[input_view, gt_spatial_view, pred_spatial_view, gt_freq_view, pred_freq_view]
		)

		self.run.log(
			{
				f"{split}/visual_spatial": wandb.Image(spatial_grid),
				f"{split}/visual_frequency": wandb.Image(freq_grid),
				f"{split}/visual_full": wandb.Image(full_grid),
			},
			step=epoch if step is None else step,
		)

	def log_gate_stats(
		self,
		gates: torch.Tensor,
		epoch: int,
		split: str,
		*,
		step: int | None = None,
	) -> None:
		if not self.enabled:
			return

		self.run.log(
			{
				f"{split}/gate_mean": gates.mean().item(),
				f"{split}/gate_std": gates.std().item(),
				f"{split}/gate_min": gates.min().item(),
				f"{split}/gate_max": gates.max().item(),
			},
			step=epoch if step is None else step,
		)

	def log_test_results(self, metrics_dict: Dict[str, float], *, step: int | None = None) -> None:
		if not self.enabled:
			return
		payload = {f"test/{k}": v for k, v in metrics_dict.items()}
		self.run.log(payload, step=step)

	def save_checkpoint(self, model: nn.Module, epoch: int, metric_value: float, is_best: bool = False) -> None:
		state = {
			"epoch": epoch,
			"metric_value": metric_value,
			"model_state": model.state_dict(),
		}
		last_path = self.checkpoint_dir / "last.pt"
		torch.save(state, last_path)
		if is_best:
			best_path = self.checkpoint_dir / "best.pt"
			torch.save(state, best_path)

	def finish(self) -> None:
		if not self.enabled:
			return
		self.run.finish()


if __name__ == "__main__":
	# Example usage
	config = {
		"visual_samples": 6,
		"lambda_mag": 0.5,
		"mag_loss": "l1",
	}

	logger = WandbLogger(project_name="deepfake-dualstream", config=config)

	# Pseudo training loop
	for epoch in range(1, 6):
		# losses_dict should include all required per-step keys
		logger.log_train_step(
			{
				"loss_total": 1.0,
				"loss_spatial_cos": 0.2,
				"loss_spatial_mag": 0.1,
				"loss_freq_cos": 0.2,
				"loss_freq_mag": 0.1,
				"loss_consistency": 0.2,
				"loss_cls": 0.2,
			},
			lr=1e-4,
			grad_norm=0.5,
			step=epoch * 100,
		)

		val_metrics = {
			"acc": 0.9,
			"auc": 0.95,
			"f1": 0.88,
			"precision": 0.9,
			"recall": 0.86,
			"cos_sim_spatial": 0.8,
			"cos_sim_freq": 0.82,
			"dice_spatial": 0.75,
			"dice_freq": 0.76,
			"iou_spatial": 0.62,
			"iou_freq": 0.63,
			"mae_spatial": 0.08,
			"mae_freq": 0.07,
		}
		logger.log_epoch_metrics(val_metrics, epoch=epoch, split="val")

		best = val_metrics["auc"] >= 0.95
		logger.save_checkpoint(model=nn.Identity(), epoch=epoch, metric_value=val_metrics["auc"], is_best=best)

	logger.finish()
