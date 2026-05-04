import os
import glob
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import pytorch_lightning as pl

def compute_fft(img_tensor):
    """Computes 2D FFT and shifts the zero-frequency component to the center."""
    if img_tensor.dtype == torch.uint8:
        img_tensor = img_tensor.float() / 255.0
    fft_tensor = torch.fft.fft2(img_tensor, norm="ortho")
    # For model input, we usually want real-valued magnitudes or real/complex concatenated.
    # To keep it simple and match standard CNNs, we take the magnitude (log scale is often better, but abs is okay).
    fft_shifted = torch.fft.fftshift(fft_tensor)
    magnitude = torch.abs(fft_shifted)
    return magnitude

class DeepfakeProcessedDataset(Dataset):
    def __init__(self, data_dir, split_mode="train"):
        self.data_dir = Path(data_dir)
        self.files = []
        
        # Discover all processed jpegs
        all_images = glob.glob(str(self.data_dir / "**" / "*.jpg"), recursive=True)
        all_images = [Path(p) for p in all_images]
        
        # Simple deterministic split based on filenames to prevent data leakage
        all_images = sorted(all_images)
        total_len = len(all_images)
        
        # Split: 70% Train, 15% Validation, 15% Test
        train_end = int(total_len * 0.70)
        val_end = int(total_len * 0.85)
        
        if split_mode == "train":
            self.files = all_images[:train_end]
        elif split_mode == "val":
            self.files = all_images[train_end:val_end]
        elif split_mode == "test":
            self.files = all_images[val_end:]
        else:
            raise ValueError("split_mode must be 'train', 'val', or 'test'")
            
        print(f"Loaded {len(self.files)} samples for {split_mode} phase.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        
        # 1. Read Spatial Image
        img_spatial = read_image(str(img_path))
        img_spatial_float = img_spatial.float() / 255.0
        
        # Determine Label and Load Masks
        is_real = "Original" in img_path.parts
        label = 0.0 if is_real else 1.0
        
        base_dir = img_path.parent
        base_name = img_path.stem
        
        # Load Precomputed FFT representation
        # It's faster to read this from disk than calculating it dynamically
        fake_fft_path = base_dir / f"{base_name}_fake_fft.pt"
        real_fft_path = base_dir / f"{base_name}_fft.pt"
        
        try:
            # If the script successfully saved the FFT tensor earlier
            if fake_fft_path.exists():
                img_freq_complex = torch.load(fake_fft_path, weights_only=True)
            else:
                img_freq_complex = torch.load(real_fft_path, weights_only=True)
            img_freq = torch.abs(img_freq_complex)  # ResNet expects real-valued float magnitudes
        except Exception:
            # Fallback if fft doesn't exist (e.g. for Original images before running latest data processor)
            img_freq = compute_fft(img_spatial_float)
            
        # VERY IMPORTANT: Normalize frequency inputs so they don't blow up the ResNet
        # FFT max magnitudes can reach 1,000,000+. We must log-compress them for CNN ingestion.
        img_freq = torch.log1p(img_freq)
            
            
        # PyTorch default collate needs same-shaped tensors in a batch.
        if not is_real:
            # Fake: Load the precomputed ground truth masks
            base_dir = img_path.parent
            base_name = img_path.stem
            
            spatial_mask_path = base_dir / f"{base_name}_spatial_mask.pt"
            freq_mask_path = base_dir / f"{base_name}_freq_mask.pt"
            
            try:
                clue_mask_gt = torch.load(spatial_mask_path, weights_only=True)
                fft_clue_gt = torch.load(freq_mask_path, weights_only=True)
            except Exception:
                # Fallback if a mask failed to save during processing
                clue_mask_gt = torch.zeros((3, 480, 480))
                fft_clue_gt = torch.zeros((3, 480, 480))
        else:
            # Real: No masks needed (model logic ignores them), but we return zeros for batch collation
            clue_mask_gt = torch.zeros((3, 480, 480))
            fft_clue_gt = torch.zeros((3, 480, 480))

        return {
            'img_spatial': img_spatial_float,
            'img_freq': img_freq,
            'label': torch.tensor(label, dtype=torch.float32),
            'clue_mask_gt': clue_mask_gt.float(),
            'fft_clue_gt': fft_clue_gt.float()
        }

class DeepfakeDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data/processed/FF32_extractedFrames", batch_size=16, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DeepfakeProcessedDataset(self.data_dir, split_mode="train")
            self.val_dataset = DeepfakeProcessedDataset(self.data_dir, split_mode="val")
        if stage == 'test' or stage is None:
            self.test_dataset = DeepfakeProcessedDataset(self.data_dir, split_mode="test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, drop_last=True, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=True)
