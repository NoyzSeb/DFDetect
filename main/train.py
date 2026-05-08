from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import optuna
from optuna.trial import TrialState
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.append(str(ROOT_DIR))

from main.efNetStructure import DualEfficientNet
from main.uNetStructure import DualUNet
from utils.inputProc import extract_spatial_freq_masks
from utils.ff32_dataset import FF32FramesDataset, SplitRatios
from utils.lossFunc import GatedFeatureFusion, unet_mask_losses
from utils.wandb_logger import (
	WandbLogger,
	compute_cosine_sim,
	compute_dice,
	compute_iou,
)


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
	"""Create train/val dataloaders.

	Expected batch item keys: input_img, freq_img, gtruth_img, label.
	"""
	raw_root = Path(args.dataset_root)
	ratios = SplitRatios(train=args.split_train, val=args.split_val, test=args.split_test)
	train_ds = FF32FramesDataset(
		raw_root,
		split="train",
		split_ratios=ratios,
		seed=args.split_seed,
	)
	val_ds = FF32FramesDataset(
		raw_root,
		split="val",
		split_ratios=ratios,
		seed=args.split_seed,
	)
	test_ds = FF32FramesDataset(
		raw_root,
		split="test",
		split_ratios=ratios,
		seed=args.split_seed,
	)

	train_loader = DataLoader(
		train_ds,
		batch_size=args.batch_size,
		shuffle=args.shuffle,
		num_workers=args.num_workers,
		pin_memory=args.device == "cuda",
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=args.device == "cuda",
	)
	test_loader = DataLoader(
		test_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=args.device == "cuda",
	)
	return train_loader, val_loader, test_loader


def build_model(args: argparse.Namespace) -> nn.Module:
	"""Create the dual-stream model.

	Model should return: logits, pred_spatial_mask, pred_freq_mask, gates(optional).
	"""
	backbone = DualEfficientNet(
		spatial_in_channels=args.spatial_in_channels,
		freq_in_channels=args.freq_in_channels,
		version=args.efnet_version,
		pretrained=args.efnet_pretrained,
		device=args.device,
	)

	with torch.no_grad():
		dummy = torch.zeros(
			1,
			args.spatial_in_channels,
			args.image_size,
			args.image_size,
			device=backbone.device,
		)
		spat_map, freq_map = backbone(dummy, dummy)
		map_channels = spat_map.shape[1]

	unet = DualUNet(
		spatial_in_channels=map_channels,
		freq_in_channels=map_channels,
		spatial_out_channels=map_channels,
		freq_out_channels=map_channels,
		base_channels=args.unet_base_channels,
	)

	fusion = GatedFeatureFusion(map_channels) if args.use_gating else None
	num_classes = 1 if args.cls_loss == "bce" else 2

	class DualStreamModel(nn.Module):
		def __init__(self) -> None:
			super().__init__()
			self.backbone = backbone
			self.unet = unet
			self.fusion = fusion
			self.pool = nn.AdaptiveAvgPool2d(1)
			self.classifier = nn.Linear(map_channels, num_classes)
			self.spat_mask_head = nn.Conv2d(map_channels, 3, kernel_size=1)
			self.freq_mask_head = nn.Conv2d(map_channels, 3, kernel_size=1)

		def forward(
			self,
			spatial_x: torch.Tensor,
			freq_x: torch.Tensor,
		) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
			spat_map, freq_map = self.backbone(spatial_x, freq_x)
			if self.fusion is not None:
				fused, gates = self.fusion(spat_map, freq_map)
			else:
				fused, gates = spat_map, None

			logits = self.classifier(self.pool(fused).flatten(1))
			pred_spatial, pred_freq = self.unet(spat_map, freq_map)
			pred_spatial = self.spat_mask_head(pred_spatial)
			pred_freq = self.freq_mask_head(pred_freq)
			pred_spatial = F.interpolate(
				pred_spatial,
				size=spatial_x.shape[-2:],
				mode="bilinear",
				align_corners=False,
			)
			pred_freq = F.interpolate(
				pred_freq,
				size=spatial_x.shape[-2:],
				mode="bilinear",
				align_corners=False,
			)
			return logits, pred_spatial, pred_freq, gates

	return DualStreamModel()


def _classification_loss(logits: torch.Tensor, labels: torch.Tensor, loss_type: str) -> torch.Tensor:
	if loss_type == "bce":
		labels = labels.float()
		return F.binary_cross_entropy_with_logits(logits.squeeze(1), labels)
	if loss_type == "ce":
		return F.cross_entropy(logits, labels.long())
	raise ValueError(f"Unsupported cls_loss: {loss_type}")


def _calibrated_components(
	pred: torch.Tensor,
	target: torch.Tensor,
	*,
	mag_loss: str,
	eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
	cos = compute_cosine_sim(pred, target, eps=eps).mean()
	loss_cos = 1 - cos
	if mag_loss == "l1":
		loss_mag = F.l1_loss(pred, target)
	elif mag_loss == "l2":
		loss_mag = F.mse_loss(pred, target)
	else:
		raise ValueError(f"Unsupported mag_loss: {mag_loss}")
	return loss_cos, loss_mag


def _mask_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
	return {
		"cos": compute_cosine_sim(pred, target).mean().item(),
		"dice": compute_dice(pred, target).mean().item(),
		"iou": compute_iou(pred, target).mean().item(),
		"mae": F.l1_loss(pred, target).item(),
	}


def _classification_metrics(y_true: Iterable[int], y_prob: Iterable[float]) -> Dict[str, float]:
	y_true = np.asarray(list(y_true))
	y_prob = np.asarray(list(y_prob))
	y_pred = (y_prob >= 0.5).astype(int)
	return {
		"acc": accuracy_score(y_true, y_pred),
		"auc": roc_auc_score(y_true, y_prob),
		"f1": f1_score(y_true, y_pred),
		"precision": precision_score(y_true, y_pred),
		"recall": recall_score(y_true, y_pred),
		"y_true": y_true.tolist(),
		"y_pred": y_pred.tolist(),
		"class_names": ["real", "fake"],
	}


def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	logger: WandbLogger,
	epoch: int,
	args: argparse.Namespace,
) -> None:
	model.train()
	pbar = tqdm(loader, desc=f"train epoch {epoch}", dynamic_ncols=True)
	base_step = (epoch - 1) * len(loader)
	for step, batch in enumerate(pbar, start=1):
		input_img = batch["input_img"].to(device)
		freq_img = batch["freq_img"].to(device)
		gtruth_img = batch["gtruth_img"].to(device)
		labels = batch["label"].to(device)

		optimizer.zero_grad(set_to_none=True)

		logits, pred_spatial, pred_freq, gates = model(input_img, freq_img)
		spatial_mask, freq_mask = extract_spatial_freq_masks(gtruth_img, input_img)

		loss_spatial, loss_freq, loss_cons = unet_mask_losses(
			spatial_mask,
			freq_mask,
			pred_spatial,
			pred_freq,
			lambda_mag=args.lambda_mag,
			mag_loss=args.mag_loss,
		)
		loss_cls = _classification_loss(logits, labels, args.cls_loss)
		loss_total = loss_spatial + loss_freq + loss_cons + args.cls_weight * loss_cls

		loss_total.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
		optimizer.step()

		loss_spatial_cos, loss_spatial_mag = _calibrated_components(
			pred_spatial, spatial_mask, mag_loss=args.mag_loss
		)
		loss_freq_cos, loss_freq_mag = _calibrated_components(
			pred_freq, freq_mask, mag_loss=args.mag_loss
		)

		if step % args.log_every == 0:
			logger.log_train_step(
				{
					"loss_total": loss_total.item(),
					"loss_spatial_cos": loss_spatial_cos.item(),
					"loss_spatial_mag": loss_spatial_mag.item(),
					"loss_freq_cos": loss_freq_cos.item(),
					"loss_freq_mag": loss_freq_mag.item(),
					"loss_consistency": loss_cons.item(),
					"loss_cls": loss_cls.item(),
				},
				lr=optimizer.param_groups[0]["lr"],
				grad_norm=float(grad_norm),
				step=base_step + step,
			)

			if gates is not None:
				logger.log_gate_stats(gates.detach(), epoch=epoch, split="train", step=base_step + step)

		pbar.set_postfix({"loss": f"{loss_total.item():.4f}"})
		if args.max_steps and step >= args.max_steps:
			break


@torch.no_grad()
def evaluate(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	logger: WandbLogger,
	epoch: int,
	split: str,
	args: argparse.Namespace,
	train_steps_per_epoch: int,
) -> Dict[str, float]:
	model.eval()
	global_step = epoch * train_steps_per_epoch
	y_true = []
	y_prob = []
	if not hasattr(logger, "_fixed_visual_batch"):
		logger._fixed_visual_batch = None
	mask_stats = {
		"spatial": {"cos": [], "dice": [], "iou": [], "mae": []},
		"freq": {"cos": [], "dice": [], "iou": [], "mae": []},
	}
	loss_stats = {
		"loss_total": [],
		"loss_spatial": [],
		"loss_freq": [],
		"loss_consistency": [],
		"loss_cls": [],
		"loss_spatial_cos": [],
		"loss_spatial_mag": [],
		"loss_freq_cos": [],
		"loss_freq_mag": [],
	}

	for step, batch in enumerate(tqdm(loader, desc=f"{split} epoch {epoch}", dynamic_ncols=True), start=1):
		input_img = batch["input_img"].to(device)
		freq_img = batch["freq_img"].to(device)
		gtruth_img = batch["gtruth_img"].to(device)
		labels = batch["label"].to(device)

		logits, pred_spatial, pred_freq, gates = model(input_img, freq_img)
		probs = torch.sigmoid(logits.squeeze(1)) if args.cls_loss == "bce" else F.softmax(logits, dim=1)[:, 1]

		spatial_mask, freq_mask = extract_spatial_freq_masks(gtruth_img, input_img)
		if logger._fixed_visual_batch is None:
			real_idx = (labels == 0).nonzero(as_tuple=True)[0]
			fake_idx = (labels == 1).nonzero(as_tuple=True)[0]
			if real_idx.numel() > 0 and fake_idx.numel() > 0:
				indices = torch.stack([real_idx[0], fake_idx[0]])
				logger._fixed_visual_batch = {
					"input_img": input_img[indices].detach().cpu(),
					"gt_spatial": spatial_mask[indices].detach().cpu(),
					"gt_freq": freq_mask[indices].detach().cpu(),
					"pred_spatial": pred_spatial[indices].detach().cpu(),
					"pred_freq": pred_freq[indices].detach().cpu(),
				}
		spatial_metrics = _mask_metrics(pred_spatial, spatial_mask)
		freq_metrics = _mask_metrics(pred_freq, freq_mask)

		loss_spatial, loss_freq, loss_cons = unet_mask_losses(
			spatial_mask,
			freq_mask,
			pred_spatial,
			pred_freq,
			lambda_mag=args.lambda_mag,
			mag_loss=args.mag_loss,
		)
		loss_cls = _classification_loss(logits, labels, args.cls_loss)
		loss_total = loss_spatial + loss_freq + loss_cons + args.cls_weight * loss_cls
		loss_spatial_cos, loss_spatial_mag = _calibrated_components(
			pred_spatial, spatial_mask, mag_loss=args.mag_loss
		)
		loss_freq_cos, loss_freq_mag = _calibrated_components(
			pred_freq, freq_mask, mag_loss=args.mag_loss
		)

		loss_stats["loss_total"].append(loss_total.item())
		loss_stats["loss_spatial"].append(loss_spatial.item())
		loss_stats["loss_freq"].append(loss_freq.item())
		loss_stats["loss_consistency"].append(loss_cons.item())
		loss_stats["loss_cls"].append(loss_cls.item())
		loss_stats["loss_spatial_cos"].append(loss_spatial_cos.item())
		loss_stats["loss_spatial_mag"].append(loss_spatial_mag.item())
		loss_stats["loss_freq_cos"].append(loss_freq_cos.item())
		loss_stats["loss_freq_mag"].append(loss_freq_mag.item())

		for key in mask_stats["spatial"]:
			mask_stats["spatial"][key].append(spatial_metrics[key])
			mask_stats["freq"][key].append(freq_metrics[key])

		y_true.extend(labels.detach().cpu().tolist())
		y_prob.extend(probs.detach().cpu().tolist())
		if args.val_max_steps and step >= args.val_max_steps:
			break

	metrics = _classification_metrics(y_true, y_prob)
	metrics.update(
		{
			"cos_sim_spatial": float(np.mean(mask_stats["spatial"]["cos"])),
			"cos_sim_freq": float(np.mean(mask_stats["freq"]["cos"])),
			"dice_spatial": float(np.mean(mask_stats["spatial"]["dice"])),
			"dice_freq": float(np.mean(mask_stats["freq"]["dice"])),
			"iou_spatial": float(np.mean(mask_stats["spatial"]["iou"])),
			"iou_freq": float(np.mean(mask_stats["freq"]["iou"])),
			"mae_spatial": float(np.mean(mask_stats["spatial"]["mae"])),
			"mae_freq": float(np.mean(mask_stats["freq"]["mae"])),
			"loss_total": float(np.mean(loss_stats["loss_total"])),
			"loss_spatial": float(np.mean(loss_stats["loss_spatial"])),
			"loss_freq": float(np.mean(loss_stats["loss_freq"])),
			"loss_consistency": float(np.mean(loss_stats["loss_consistency"])),
			"loss_cls": float(np.mean(loss_stats["loss_cls"])),
			"loss_spatial_cos": float(np.mean(loss_stats["loss_spatial_cos"])),
			"loss_spatial_mag": float(np.mean(loss_stats["loss_spatial_mag"])),
			"loss_freq_cos": float(np.mean(loss_stats["loss_freq_cos"])),
			"loss_freq_mag": float(np.mean(loss_stats["loss_freq_mag"])),
		}
	)

	logger.log_epoch_metrics(metrics, epoch=epoch, split=split, step=global_step)
	if logger._fixed_visual_batch is not None:
		logger.log_visuals(
			logger._fixed_visual_batch["input_img"],
			logger._fixed_visual_batch["gt_spatial"],
			logger._fixed_visual_batch["gt_freq"],
			logger._fixed_visual_batch["pred_spatial"],
			logger._fixed_visual_batch["pred_freq"],
			epoch=epoch,
			split=split,
			step=global_step,
		)
	return metrics


def train_and_validate(args: argparse.Namespace) -> float:
	device = torch.device(args.device)
	if device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA is required but not available. Set --device cpu to run on CPU.")
	set_seed(args.seed)

	train_loader, val_loader, test_loader = build_dataloaders(args)
	model = build_model(args).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = None
	if args.lr_scheduler == "cosine":
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	elif args.lr_scheduler == "step":
		scheduler = torch.optim.lr_scheduler.StepLR(
			optimizer,
			step_size=args.lr_step_size,
			gamma=args.lr_gamma,
		)

	config = {
		"batch_size": args.batch_size,
		"epochs": args.epochs,
		"lr": args.lr,
		"lr_scheduler": args.lr_scheduler,
		"lr_step_size": args.lr_step_size,
		"lr_gamma": args.lr_gamma,
		"weight_decay": args.weight_decay,
		"lambda_mag": args.lambda_mag,
		"mag_loss": args.mag_loss,
		"cls_weight": args.cls_weight,
		"cls_loss": args.cls_loss,
		"efnet_version": args.efnet_version,
		"efnet_pretrained": args.efnet_pretrained,
		"unet_base_channels": args.unet_base_channels,
		"use_gating": args.use_gating,
		"image_size": args.image_size,
		"spatial_in_channels": args.spatial_in_channels,
		"freq_in_channels": args.freq_in_channels,
		"visual_samples": args.visual_samples,
	}
	logger = WandbLogger(project_name=args.project_name, config=config, run_name=args.run_name)

	log_path = logger.checkpoint_dir / "log.txt"
	log_path.parent.mkdir(parents=True, exist_ok=True)
	file_handler = logging.FileHandler(log_path, encoding="utf-8")
	logging.basicConfig(level=logging.INFO, handlers=[file_handler, logging.StreamHandler()])
	logging.info("run_name=%s", logger.run_name)
	for key, value in config.items():
		logging.info("%s=%s", key, value)

	best_auc = -1.0
	train_steps_per_epoch = len(train_loader)
	for epoch in range(1, args.epochs + 1):
		train_one_epoch(model, train_loader, optimizer, device, logger, epoch, args)
		metrics = evaluate(
			model,
			val_loader,
			device,
			logger,
			epoch,
			"val",
			args,
			train_steps_per_epoch,
		)
		if scheduler is not None:
			scheduler.step()
		auc = metrics["auc"]
		is_best = auc > best_auc
		if is_best:
			best_auc = auc
		logger.save_checkpoint(model, epoch=epoch, metric_value=auc, is_best=is_best)

	final_test_metrics = evaluate(
		model,
		test_loader,
		device,
		logger,
		args.epochs,
		"test",
		args,
		train_steps_per_epoch,
	)
	logger.log_test_results(final_test_metrics, step=args.epochs * train_steps_per_epoch)

	logger.finish()
	return best_auc


def sanity_check(args: argparse.Namespace) -> None:
	device = torch.device(args.device)
	if device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA is required but not available. Set --device cpu to run on CPU.")
	set_seed(args.seed)

	train_loader, _, _ = build_dataloaders(args)
	model = build_model(args).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	batch = next(iter(train_loader))
	input_img = batch["input_img"].to(device)
	freq_img = batch["freq_img"].to(device)
	gtruth_img = batch["gtruth_img"].to(device)
	labels = batch["label"].to(device)

	model.train()
	optimizer.zero_grad(set_to_none=True)
	logits, pred_spatial, pred_freq, _ = model(input_img, freq_img)
	spatial_mask, freq_mask = extract_spatial_freq_masks(gtruth_img, input_img)

	loss_spatial, loss_freq, loss_cons = unet_mask_losses(
		spatial_mask,
		freq_mask,
		pred_spatial,
		pred_freq,
		lambda_mag=args.lambda_mag,
		mag_loss=args.mag_loss,
	)
	loss_cls = _classification_loss(logits, labels, args.cls_loss)
	loss_total = loss_spatial + loss_freq + loss_cons + args.cls_weight * loss_cls

	loss_total.backward()
	optimizer.step()

	print(
		"Sanity check passed. "
		f"loss_total={loss_total.item():.4f}, "
		f"loss_cls={loss_cls.item():.4f}"
	)


def run_optuna(args: argparse.Namespace) -> None:
	def objective(trial: optuna.Trial) -> float:
		print(f"[optuna] trial {trial.number} starting")
		args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
		args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
		args.lambda_mag = trial.suggest_float("lambda_mag", 0.1, 1.0)
		args.mag_loss = trial.suggest_categorical("mag_loss", ["l1", "l2"])
		args.cls_weight = trial.suggest_float("cls_weight", 0.5, 2.0)
		args.epochs = args.optuna_epochs
		args.max_steps = args.optuna_max_steps
		args.val_max_steps = args.optuna_val_max_steps
		score = train_and_validate(args)
		print(f"[optuna] trial {trial.number} done: auc={score:.4f}")
		return score

	def on_trial_complete(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
		if trial.state != TrialState.COMPLETE:
			return
		status = "new best" if study.best_trial.number == trial.number else "completed"
		print(
			f"[optuna] trial {trial.number} {status}: value={trial.value:.4f} | "
			f"best={study.best_value:.4f} (trial {study.best_trial.number})"
		)

	study = optuna.create_study(direction="maximize")
	study.optimize(objective, n_trials=args.optuna_trials, callbacks=[on_trial_complete])

	print("Best trial:")
	print(study.best_trial.params)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Training script for dual-stream deepfake model.")
	parser.add_argument("--config", type=str, default="config.json", help="Path to JSON config file.")
	parser.add_argument(
		"--dataset-root",
		type=str,
		default="data/processed/FF32_extractedFrames",
		help="Root folder with Original and manipulated subfolders.",
	)
	parser.add_argument("--split-train", type=float, default=0.8)
	parser.add_argument("--split-val", type=float, default=0.1)
	parser.add_argument("--split-test", type=float, default=0.1)
	parser.add_argument("--image-size", type=int, default=480)
	parser.add_argument("--spatial-in-channels", type=int, default=3)
	parser.add_argument("--freq-in-channels", type=int, default=3)
	parser.add_argument("--split-seed", type=int, default=42)
	parser.add_argument("--shuffle", action="store_true", default=True)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--lr-scheduler", type=str, choices=["none", "cosine", "step"], default="none")
	parser.add_argument("--lr-step-size", type=int, default=5)
	parser.add_argument("--lr-gamma", type=float, default=0.5)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--log-every", type=int, default=10)
	parser.add_argument("--max-grad-norm", type=float, default=1.0)
	parser.add_argument("--project-name", type=str, default="deepfake-dualstream")
	parser.add_argument("--run-name", type=str, default=None)
	parser.add_argument("--visual-samples", type=int, default=6)
	parser.add_argument("--lambda-mag", type=float, default=0.5)
	parser.add_argument("--mag-loss", type=str, choices=["l1", "l2"], default="l1")
	parser.add_argument("--cls-weight", type=float, default=1.0)
	parser.add_argument("--cls-loss", type=str, choices=["bce", "ce"], default="bce")
	parser.add_argument("--efnet-version", type=str, default="b0")
	parser.add_argument("--no-efnet-pretrained", dest="efnet_pretrained", action="store_false")
	parser.set_defaults(efnet_pretrained=True)
	parser.add_argument("--unet-base-channels", type=int, default=32)
	parser.add_argument("--no-gating", dest="use_gating", action="store_false")
	parser.set_defaults(use_gating=True)
	parser.add_argument("--optuna-trials", type=int, default=0)
	parser.add_argument("--optuna-epochs", type=int, default=5)
	parser.add_argument("--optuna-max-steps", type=int, default=0)
	parser.add_argument("--optuna-val-max-steps", type=int, default=0)
	parser.add_argument("--max-steps", type=int, default=0)
	parser.add_argument("--val-max-steps", type=int, default=0)
	parser.add_argument("--sanity", action="store_true", help="Run a single-batch smoke test and exit.")
	args = parser.parse_args()

	config_path = Path(args.config)
	if config_path.exists():
		with config_path.open("r", encoding="utf-8") as handle:
			config_data = json.load(handle)
		for key, value in config_data.items():
			if hasattr(args, key):
				setattr(args, key, value)
	return args


def main() -> None:
	args = parse_args()
	if args.sanity:
		sanity_check(args)
		return
	if args.optuna_trials > 0:
		run_optuna(args)
	else:
		train_and_validate(args)


if __name__ == "__main__":
	main()
