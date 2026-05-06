from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def apply_fft2(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
	if image.dim() < 2:
		raise ValueError("Input image must have at least 2 dimensions (H, W).")

	spat = image
	fft = torch.fft.fft2(image, dim=(-2, -1))
	fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
	freq = torch.abs(fft_shifted)
	freq = torch.log1p(freq)
	reduce_dims = tuple(range(1, freq.dim() - 2)) + (-2, -1) if freq.dim() > 2 else (-2, -1)
	freq = freq / (freq.mean(dim=reduce_dims, keepdim=True) + 1e-8)
	return spat, freq


def extract_spatial_freq_masks(
	gtruth: torch.Tensor,
	input_img: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
	if gtruth.shape != input_img.shape:
		raise ValueError("gtruth and input_img must have the same shape.")

	spat_gtruth, freq_gtruth = apply_fft2(gtruth)
	spat_input, freq_input = apply_fft2(input_img)

	spatial_mask = spat_input - spat_gtruth
	freq_mask = freq_input - freq_gtruth
	return spatial_mask, freq_mask


def cosine_similarity_masks(mask_a: torch.Tensor, mask_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
	if mask_a.shape != mask_b.shape:
		raise ValueError("mask_a and mask_b must have the same shape.")

	if mask_a.dim() == 3:
		mask_a = mask_a.unsqueeze(0)
		mask_b = mask_b.unsqueeze(0)
	elif mask_a.dim() < 3:
		raise ValueError("Masks must have shape (C, H, W) or (N, C, H, W).")

	flat_a = mask_a.flatten(1)
	flat_b = mask_b.flatten(1)
	return F.cosine_similarity(flat_a, flat_b, dim=1, eps=eps)


def cosine_similarity_spat_freq(
	spat_mask_a: torch.Tensor,
	spat_mask_b: torch.Tensor,
	freq_mask_a: torch.Tensor,
	freq_mask_b: torch.Tensor,
	eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
	spat_score = cosine_similarity_masks(spat_mask_a, spat_mask_b, eps=eps)
	freq_score = cosine_similarity_masks(freq_mask_a, freq_mask_b, eps=eps)
	return spat_score, freq_score
