from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torchvision import models


def _set_first_two_convs_same_padding(module: nn.Module) -> None:
	convs = []
	for submodule in module.modules():
		if isinstance(submodule, nn.Conv2d):
			convs.append(submodule)
			if len(convs) == 2:
				break

	for conv in convs:
		k_h, k_w = conv.kernel_size
		conv.padding = (k_h // 2, k_w // 2)
		conv.stride = (1, 1)


def _make_efficientnet(version: str, in_channels: int, pretrained: bool) -> nn.Module:
	version = version.lower()
	factory = {
		"b0": models.efficientnet_b0,
		"b1": models.efficientnet_b1,
		"b2": models.efficientnet_b2,
		"b3": models.efficientnet_b3,
		"b4": models.efficientnet_b4,
		"b5": models.efficientnet_b5,
		"b6": models.efficientnet_b6,
		"b7": models.efficientnet_b7,
	}.get(version)

	if factory is None:
		raise ValueError(f"Unsupported EfficientNet version: {version}")

	weights = None
	if pretrained:
		weights_enum_name = f"EfficientNet_{version.upper()}_Weights"
		weights_enum = getattr(models, weights_enum_name, None)
		if weights_enum is None:
			raise ValueError(f"No weights enum found for EfficientNet {version}")
		weights = weights_enum.DEFAULT

	model = factory(weights=weights)

	if in_channels != 3:
		stem = model.features[0][0]
		model.features[0][0] = nn.Conv2d(
			in_channels=in_channels,
			out_channels=stem.out_channels,
			kernel_size=stem.kernel_size,
			stride=stem.stride,
			padding=stem.padding,
			bias=stem.bias is not None,
		)

	_set_first_two_convs_same_padding(model)
	return model


class DualEfficientNet(nn.Module):
	def __init__(
		self,
		spatial_in_channels: int = 3,
		freq_in_channels: int = 3,
		version: str = "b0",
		pretrained: bool = True,
		device: str | torch.device = "cuda",
	) -> None:
		super().__init__()
		self.spatial_backbone = _make_efficientnet(version, spatial_in_channels, pretrained)
		self.freq_backbone = _make_efficientnet(version, freq_in_channels, pretrained)
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA is required but not available on this system.")
		self.device = torch.device(device)
		self.to(self.device)

	def forward(self, spatial_x: torch.Tensor, freq_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		if spatial_x.device != self.device:
			spatial_x = spatial_x.to(self.device, non_blocking=True)
		if freq_x.device != self.device:
			freq_x = freq_x.to(self.device, non_blocking=True)
		spatMap = self.spatial_backbone.features[:2](spatial_x)
		freqMap = self.freq_backbone.features[:2](freq_x)
		return spatMap, freqMap
