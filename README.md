# DFDetect_main

This folder contains the core model components and utilities for the dual-branch spatial/frequency deepfake pipeline.

## What is implemented

- Dual EfficientNet backbone to extract early feature maps for spatial and frequency inputs.
- The first two convolution layers are forced to preserve spatial size (padding with stride 1).
- ImageNet pretrained weights are used by default.
- CUDA-only execution is enforced (raises if GPU is not available).
- Dual U-Net branches that take `spatMap` and `freqMap` separately.
- FFT-based preprocessing utilities to build spatial/frequency masks and compare them with cosine similarity.
- Gated fusion utility for spatial/frequency feature streams (attention-style gating).
- Mask loss utilities with calibrated cosine loss and a frequency-domain consistency term.
- Raw image preprocessing script that center-crops/pads to a fixed size.

## Key files

- main/efNetStructure.py
  - `DualEfficientNet` returns early feature maps (`features[:2]`) as `spatMap` and `freqMap`.
- main/uNetStructure.py
  - `DualUNet` with two independent U-Net branches.
- utils/inputProc.py
  - `apply_fft2`: returns spatial input and log-magnitude frequency (mean-normalized).
  - `extract_spatial_freq_masks`: creates spatial and frequency masks between input and ground truth.
  - `cosine_similarity_masks` and `cosine_similarity_spat_freq`: compare masks with cosine similarity.
- utils/lossFunc.py
  - `GatedFeatureFusion`: per-channel gating between spatial/frequency features.
  - `calibrated_cos_loss`: cosine loss plus L1/L2 magnitude calibration.
  - `spatial_branch_loss`, `freq_branch_loss`, `consistency_loss`, `unet_mask_losses`.
- utils/dataProc.py
  - CLI for center crop/pad to a fixed square size and writing to `data/processed`.
- utils/wandb_logger.py
  - `WandbLogger` for W&B logging, visuals, and checkpoint management.
  - Helpers: `get_next_run_name`, `compute_dice`, `compute_iou`, `compute_cosine_sim`.
- main/train.py
  - Training scaffold with Optuna hyperparameter search, W&B logging, and CLI args.

## Notes

- Frequency output uses log-magnitude and per-sample mean normalization to stabilize cosine similarity.
- If you need a different early cutoff or a different EfficientNet version, adjust the `version` or slice in `DualEfficientNet`.

## Minimal usage sketch

```python
import torch
from main.efNetStructure import DualEfficientNet
from main.uNetStructure import DualUNet
from utils.inputProc import extract_spatial_freq_masks, cosine_similarity_spat_freq
from utils.lossFunc import unet_mask_losses

backbone = DualEfficientNet(version="b0")
spatMap, freqMap = backbone(spatial_x, freq_x)

unet = DualUNet(
    spatial_in_channels=spatMap.shape[1],
    freq_in_channels=freqMap.shape[1],
    spatial_out_channels=spatMap.shape[1],
    freq_out_channels=freqMap.shape[1],
)
spat_pred, freq_pred = unet(spatMap, freqMap)

spat_mask, freq_mask = extract_spatial_freq_masks(gtruth, input_img)
spat_score, freq_score = cosine_similarity_spat_freq(spat_mask, spat_pred, freq_mask, freq_pred)

loss_spat, loss_freq, loss_cons = unet_mask_losses(
  spat_mask,
  freq_mask,
  spat_pred,
  freq_pred,
  lambda_mag=0.5,
  mag_loss="l1",
)

total_loss = loss_spat + loss_freq + loss_cons
```

## Data preprocessing

Center crop/pad raw images to a fixed size and preserve folder structure:

```bash
python utils/dataProc.py --raw-root data/raw --processed-root data/processed --size 480
```

## Training

The training script is wired to the FF32_extractedFrames dataset and the dual-stream model.

Run a standard training job:

```bash
python main/train.py --batch-size 8 --epochs 20
```

Run Optuna search (uses `--optuna-epochs` per trial):

```bash
python main/train.py --optuna-trials 10 --optuna-epochs 5
```
