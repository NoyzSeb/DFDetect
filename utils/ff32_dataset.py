from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .inputProc import apply_fft2, is_real_filename, resolve_original_from_processed


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SplitRatios:
	train: float = 0.8
	val: float = 0.1
	test: float = 0.1

	def validate(self) -> None:
		total = self.train + self.val + self.test
		if abs(total - 1.0) > 1e-6:
			raise ValueError("Split ratios must sum to 1.0")


class FF32FramesDataset(Dataset):
	"""FF32_extractedFrames dataset with real/fake labeling and original mapping."""

	def __init__(
		self,
		root: Path,
		split: str,
		*,
		split_ratios: SplitRatios,
		seed: int = 42,
		return_freq: bool = True,
		transform: transforms.Compose | None = None,
	) -> None:
		self.root = root
		self.split = split
		split_ratios.validate()

		self._transform = transform or transforms.ToTensor()
		self._return_freq = return_freq
		self._originals_root = root / "Original"

		paths = self._gather_paths(root)
		self._paths = self._split_paths(paths, split, split_ratios, seed)

	def _gather_paths(self, root: Path) -> List[Path]:
		paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
		return sorted(paths)

	def _split_paths(
		self,
		paths: List[Path],
		split: str,
		ratios: SplitRatios,
		seed: int,
	) -> List[Path]:
		if split not in {"train", "val", "test"}:
			raise ValueError("split must be train, val, or test")

		g = torch.Generator().manual_seed(seed)
		indices = torch.randperm(len(paths), generator=g).tolist()
		paths = [paths[i] for i in indices]

		n_total = len(paths)
		n_train = int(n_total * ratios.train)
		n_val = int(n_total * ratios.val)
		n_test = n_total - n_train - n_val

		if split == "train":
			return paths[:n_train]
		if split == "val":
			return paths[n_train : n_train + n_val]
		return paths[n_train + n_val : n_train + n_val + n_test]

	def __len__(self) -> int:
		return len(self._paths)

	def _load_image(self, path: Path) -> torch.Tensor:
		with Image.open(path) as img:
			img = img.convert("RGB")
			return self._transform(img)

	def _is_real_path(self, path: Path) -> bool:
		if "Original" in path.parts:
			return True
		return is_real_filename(path.name)

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
		path = self._paths[idx]
		is_real = self._is_real_path(path)
		input_img = self._load_image(path)

		if is_real:
			gtruth_img = input_img.clone()
			label = torch.tensor(0, dtype=torch.long)
		else:
			orig_path = resolve_original_from_processed(path, self._originals_root)
			gtruth_img = self._load_image(orig_path)
			label = torch.tensor(1, dtype=torch.long)

		if self._return_freq:
			_, freq_img = apply_fft2(input_img)
		else:
			freq_img = torch.empty(0)

		return {
			"input_img": input_img,
			"freq_img": freq_img,
			"gtruth_img": gtruth_img,
			"label": label,
			"path": str(path),
		}
