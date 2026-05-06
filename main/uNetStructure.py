from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class _DoubleConv(nn.Module):
	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class _Down(nn.Module):
	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			_DoubleConv(in_channels, out_channels),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class _Up(nn.Module):
	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
		self.conv = _DoubleConv(in_channels, out_channels)

	def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
		x = self.up(x)
		# Pad if needed to match spatial size from skip
		diff_y = skip.size(2) - x.size(2)
		diff_x = skip.size(3) - x.size(3)
		if diff_y != 0 or diff_x != 0:
			x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
		x = torch.cat([skip, x], dim=1)
		return self.conv(x)


class UNetBranch(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32) -> None:
		super().__init__()
		self.inc = _DoubleConv(in_channels, base_channels)
		self.down1 = _Down(base_channels, base_channels * 2)
		self.down2 = _Down(base_channels * 2, base_channels * 4)
		self.down3 = _Down(base_channels * 4, base_channels * 8)
		self.up1 = _Up(base_channels * 8, base_channels * 4)
		self.up2 = _Up(base_channels * 4, base_channels * 2)
		self.up3 = _Up(base_channels * 2, base_channels)
		self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x = self.up1(x4, x3)
		x = self.up2(x, x2)
		x = self.up3(x, x1)
		return self.outc(x)


class DualUNet(nn.Module):
	def __init__(
		self,
		spatial_in_channels: int,
		freq_in_channels: int,
		spatial_out_channels: int,
		freq_out_channels: int,
		base_channels: int = 32,
	) -> None:
		super().__init__()
		self.spatial_branch = UNetBranch(spatial_in_channels, spatial_out_channels, base_channels)
		self.freq_branch = UNetBranch(freq_in_channels, freq_out_channels, base_channels)

	def forward(self, spatMap: torch.Tensor, freqMap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		spatial_out = self.spatial_branch(spatMap)
		freq_out = self.freq_branch(freqMap)
		return spatial_out, freq_out
