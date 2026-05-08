from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .inputProc import cosine_similarity_masks


class GatedFeatureFusion(nn.Module):
	def __init__(self, channels: int, reduction: int = 4) -> None:
		super().__init__()
		hidden = max(8, (channels * 2) // reduction)
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.mlp = nn.Sequential(
			nn.Linear(channels * 2, hidden),
			nn.ReLU(inplace=True),
			nn.Linear(hidden, channels),
			nn.Sigmoid(),
		)

	def forward(self, f_spatial: torch.Tensor, f_fft: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		if f_spatial.shape != f_fft.shape:
			raise ValueError("f_spatial and f_fft must have the same shape (N, C, H, W).")

		pooled_spat = self.pool(f_spatial).flatten(1)
		pooled_fft = self.pool(f_fft).flatten(1)
		gates = self.mlp(torch.cat([pooled_spat, pooled_fft], dim=1)).view(
			f_spatial.size(0), f_spatial.size(1), 1, 1
		)
		fused = gates * f_spatial + (1 - gates) * f_fft
		return fused, gates


def _reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
	if reduction == "mean":
		return loss.mean()
	if reduction == "sum":
		return loss.sum()
	if reduction == "none":
		return loss
	raise ValueError(f"Unsupported reduction: {reduction}")


def _assert_non_negative(tensor: torch.Tensor, name: str) -> None:
	if torch.any(tensor < 0):
		raise ValueError(f"{name} must be non-negative for Dice loss.")


def calibrated_cos_loss(
	pred_mask: torch.Tensor,
	gt_mask: torch.Tensor,
	*,
	lambda_mag: float = 0.5,
	mag_loss: str = "l1",
	eps: float = 1e-8,
	reduction: str = "mean",
) -> torch.Tensor:
	cos = cosine_similarity_masks(gt_mask, pred_mask, eps=eps)
	if mag_loss == "l1":
		mag = F.l1_loss(pred_mask, gt_mask, reduction="none")
	elif mag_loss == "l2":
		mag = F.mse_loss(pred_mask, gt_mask, reduction="none")
	else:
		raise ValueError(f"Unsupported mag_loss: {mag_loss}")

	mag = mag.flatten(1).mean(dim=1)
	combined = (1 - cos) + lambda_mag * mag
	return _reduce_loss(combined, reduction)


def spatial_branch_loss(
	groundtruth_spatial_mask: torch.Tensor,
	pred_spatial_mask: torch.Tensor,
	*,
	lambda_mag: float = 0.5,
	mag_loss: str = "l1",
	eps: float = 1e-8,
	reduction: str = "mean",
) -> torch.Tensor:
	_assert_non_negative(groundtruth_spatial_mask, "groundtruth_spatial_mask")
	return calibrated_cos_loss(
		pred_spatial_mask,
		groundtruth_spatial_mask,
		lambda_mag=lambda_mag,
		mag_loss=mag_loss,
		eps=eps,
		reduction=reduction,
	)


def freq_branch_loss(
	groundtruth_freq_mask: torch.Tensor,
	pred_freq_mask: torch.Tensor,
	*,
	lambda_mag: float = 0.5,
	mag_loss: str = "l1",
	eps: float = 1e-8,
	reduction: str = "mean",
) -> torch.Tensor:
	_assert_non_negative(groundtruth_freq_mask, "groundtruth_freq_mask")
	return calibrated_cos_loss(
		pred_freq_mask,
		groundtruth_freq_mask,
		lambda_mag=lambda_mag,
		mag_loss=mag_loss,
		eps=eps,
		reduction=reduction,
	)


def consistency_loss(
	pred_spatial_mask: torch.Tensor,
	pred_freq_mask: torch.Tensor,
	*,
	eps: float = 1e-8,
	reduction: str = "mean",
) -> torch.Tensor:
	spatial_fft = torch.fft.fft2(pred_spatial_mask, dim=(-2, -1))
	spatial_mag = torch.abs(spatial_fft)
	cos = cosine_similarity_masks(spatial_mag, pred_freq_mask, eps=eps)
	return _reduce_loss(1 - cos, reduction)


def unet_mask_losses(
	groundtruth_spatial_mask: torch.Tensor,
	groundtruth_freq_mask: torch.Tensor,
	pred_spatial_mask: torch.Tensor,
	pred_freq_mask: torch.Tensor,
	*,
	lambda_mag: float = 0.5,
	mag_loss: str = "l1",
	eps: float = 1e-8,
	reduction: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	l_spatial = spatial_branch_loss(
		groundtruth_spatial_mask,
		pred_spatial_mask,
		lambda_mag=lambda_mag,
		mag_loss=mag_loss,
		eps=eps,
		reduction=reduction,
	)
	l_freq = freq_branch_loss(
		groundtruth_freq_mask,
		pred_freq_mask,
		lambda_mag=lambda_mag,
		mag_loss=mag_loss,
		eps=eps,
		reduction=reduction,
	)
	l_cons = consistency_loss(
		pred_spatial_mask,
		pred_freq_mask,
		eps=eps,
		reduction=reduction,
	)
	return l_spatial, l_freq, l_cons
