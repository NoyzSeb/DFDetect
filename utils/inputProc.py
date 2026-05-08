from __future__ import annotations

from pathlib import Path
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

	spatial_mask = torch.abs(spat_input - spat_gtruth)
	freq_mask = torch.abs(freq_input - freq_gtruth)
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


def is_real_filename(filename: str) -> bool:
	"""Return True if filename matches {id}_f{frame} pattern."""
	stem = Path(filename).stem
	if stem.count("_") != 1:
		return False
	left, right = stem.split("_", 1)
	if not left:
		return False
	if not right.startswith("f"):
		return False
	return right[1:].isdigit()


def resolve_original_path(fake_path: Path, originals_root: Path) -> Path:
	"""Map fake filename {id}_{junk}_{frame} to original {id}_f{frame}."""
	stem = fake_path.stem
	parts = stem.split("_")
	if len(parts) < 3:
		raise ValueError(f"Fake filename does not match expected pattern: {fake_path.name}")
	real_id = parts[0]
	frame = parts[-1]
	if frame.startswith("f"):
		frame = frame[1:]
	orig_name = f"{real_id}_f{frame}{fake_path.suffix}"
	return originals_root / orig_name


def extract_masks_for_input(
	input_img: torch.Tensor,
	original_img: torch.Tensor | None,
	*,
	is_real: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Return spatial/frequency masks with real/fake semantics."""
	if is_real:
		zeros = torch.zeros_like(input_img)
		return zeros, zeros
	if original_img is None:
		raise ValueError("original_img is required for fake inputs.")
	return extract_spatial_freq_masks(original_img, input_img)


FAKE_FOLDERS = {"Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"}


def is_fake_path(path: Path) -> bool:
	"""Return True if path is under a known manipulated folder."""
	return any(part in FAKE_FOLDERS for part in path.parts)


def resolve_original_from_processed(fake_path: Path, originals_root: Path) -> Path:
	"""Resolve original path for processed FF32_extractedFrames inputs."""
	if "Original" in fake_path.parts:
		return fake_path

	parts = fake_path.parts
	ff_idx = None
	for idx, part in enumerate(parts):
		if part.lower() == "ff32_extractedframes":
			ff_idx = idx
			break
	if ff_idx is None:
		raise ValueError(f"Path does not include FF32_extractedFrames: {fake_path}")
	if ff_idx + 2 >= len(parts):
		raise ValueError(f"Path is missing expected subfolders: {fake_path}")

	subpath = Path(*parts[ff_idx + 2 :])
	base = originals_root
	base.mkdir(parents=True, exist_ok=True)
	if is_real_filename(fake_path.name):
		return base / subpath
	return resolve_original_path(fake_path, base / subpath.parent)


def extract_masks_for_processed_path(
	input_path: Path,
	input_img: torch.Tensor,
	original_img: torch.Tensor | None,
	*,
	originals_root: Path,
) -> Tuple[torch.Tensor, torch.Tensor, Path | None]:
	"""Return masks for FF32_extractedFrames inputs and resolved original path."""
	if "Original" in input_path.parts:
		spat_mask, freq_mask = extract_masks_for_input(input_img, None, is_real=True)
		return spat_mask, freq_mask, None
	if not is_fake_path(input_path):
		raise ValueError(f"Unknown folder type for input: {input_path}")

	orig_path = resolve_original_from_processed(input_path, originals_root)
	spat_mask, freq_mask = extract_masks_for_input(input_img, original_img, is_real=False)
	return spat_mask, freq_mask, orig_path
