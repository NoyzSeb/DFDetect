import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

# Assuming the architecture is importable from models
from src.models.dual_resnet34 import DualResNet34

def dice_loss(pred_logits, target, smooth=1e-6):
    """
    Computes the Dice Loss.
    Requires predictions to be logits (we apply sigmoid here).
    """
    pred_probs = torch.sigmoid(pred_logits)
    
    # Flatten spatial dimensions
    pred_flat = pred_probs.view(pred_probs.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(1)
    union = pred_flat.sum(1) + target_flat.sum(1)
    
    dice_score = (2. * intersection + smooth) / (union + smooth)
    return 1. - dice_score.mean()

class DualResNet34LightningSystem(pl.LightningModule):
    def __init__(self, lr=1e-4, lambda1=1.0, lambda2=0.5, lambda3=0.5, lambda5=0.2, scheduler_patience=2):
        super().__init__()
        self.save_hyperparameters()
        self.model = DualResNet34()
        
        # Loss Functions
        self.bce_logits = nn.BCEWithLogitsLoss()
        
        # Lambdas passed from Hyperparameters
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda5 = lambda5
        self.scheduler_patience = scheduler_patience

    def get_lambda4(self, current_epoch):
        if current_epoch < 5:
            return 0.0
        elif 5 <= current_epoch < 10:
            return 0.3 * (current_epoch - 5) / 5.0
        else:
            return 0.3

    def forward(self, img_spatial, img_freq, run_decoders_override=None):
        return self.model(img_spatial, img_freq, run_decoders_override)

    def calculate_losses(self, batch, batch_idx, mode="train"):
        # Unpack batch
        img_spatial = batch['img_spatial']
        img_freq = batch['img_freq'] 
        labels = batch['label'].float().view(-1, 1)
        clue_mask_gt = batch.get('clue_mask_gt', None)
        fft_clue_gt = batch.get('fft_clue_gt', None)
        
        # 1. Forward Pass (Force decoder generation during training for losses)
        # Assuming label 1 = Fake, so we calculate mask losses on fake samples only
        outputs = self(img_spatial, img_freq, run_decoders_override=True)
        cls_logits = outputs['cls_logits']
        m_spatial_pred = outputs['spatial_mask']
        m_freq_pred = outputs['freq_mask']
        
        # Term 1: L_BCE (Classification Loss)
        l_bce = self.bce_logits(cls_logits, labels)
        
        # Initialize other losses
        l_spatial = torch.tensor(0.0, device=self.device)
        l_freq = torch.tensor(0.0, device=self.device)
        l_joint = torch.tensor(0.0, device=self.device)
        l_consistency = torch.tensor(0.0, device=self.device)
        
        # Find Fake samples (we only have GT masks for fakes usually)
        fake_idx = (labels.view(-1) == 1.0).nonzero(as_tuple=True)[0]
        
        if len(fake_idx) > 0 and clue_mask_gt is not None and fft_clue_gt is not None:
            # Filter representations for fake samples
            m_sp_fake = m_spatial_pred[fake_idx]
            m_fr_fake = m_freq_pred[fake_idx]
            mask_gt_fake = clue_mask_gt[fake_idx]
            
            # Tame massive FFT dynamic range and Min-Max normalize to strictly [0.0, 1.0] per image
            fft_gt_fake_raw = fft_clue_gt[fake_idx]
            fft_gt_fake_log = torch.log1p(fft_gt_fake_raw) # Log(1+x) smoothly compresses spikes (1,000,000 -> small number)
            
            # Per-sample min-max bounds
            fft_flat = fft_gt_fake_log.view(fft_gt_fake_log.size(0), -1)
            fft_min = fft_flat.amin(dim=1).view(-1, 1, 1, 1)
            fft_max = fft_flat.amax(dim=1).view(-1, 1, 1, 1)
            fft_gt_fake = (fft_gt_fake_log - fft_min) / (fft_max - fft_min + 1e-8)
            
            # Term 2: Spatial Masking Loss 
            bce_sp = self.bce_logits(m_sp_fake, mask_gt_fake)
            dice_sp = dice_loss(m_sp_fake, mask_gt_fake)
            l_spatial = (0.5 * bce_sp) + (0.5 * dice_sp)
            
            # Term 3: Frequency Masking Loss 
            bce_fr = self.bce_logits(m_fr_fake, fft_gt_fake)
            dice_fr = dice_loss(m_fr_fake, fft_gt_fake)
            l_freq = (0.5 * bce_fr) + (0.5 * dice_fr)
            
            # Term 4: Joint Representation Loss (Scalar Product)
            l_joint = l_spatial * l_freq
            
            # Term 5: Cross-domain Consistency Loss (Using all elements, even real? If unsupervised, 
            # we can run it on the whole batch, but applying just to fakes based on GT restriction here)
            # You can change `m_fr_fake` to `m_freq_pred` if evaluating consistency on REALs too.
            # Apply Sigmoid FIRST so iFFT operates naturally on purely positive 0.0-1.0 magnitude predictions
            m_freq_pred_probs = torch.sigmoid(m_freq_pred)
            m_freq_spatial = torch.fft.ifft2(m_freq_pred_probs, norm="ortho").abs()
            m_freq_spatial = m_freq_spatial / (m_freq_spatial.amax(dim=(-2,-1), keepdim=True) + 1e-6)
            
            # NOTE: We compare the spatial predicted logits/activations directly to the normalized iFFT.
            # Depending on if M_spatial_pred is raw logit or sigmoid, you may need torch.sigmoid(m_spatial_pred) 
            # here. Using M_spatial_pred as dictated by the formula.
            m_spatial_pred_norm = torch.sigmoid(m_spatial_pred) # Sigmoid maps raw logit to [0,1] space for L1
            l_consistency = F.l1_loss(m_spatial_pred_norm, m_freq_spatial)

        # Calculate dynamic lambda4 based on the schedule
        lambda4 = self.get_lambda4(self.current_epoch)
        
        # Compute Classification Accuracy
        preds = (cls_logits > 0.0).float()
        acc = (preds == labels).float().mean()
        
        # Total Loss Calculation
        total_loss = (self.lambda1 * l_bce) + \
                     (self.lambda2 * l_spatial) + \
                     (self.lambda3 * l_freq) + \
                     (lambda4 * l_joint) + \
                     (self.lambda5 * l_consistency)
                     
        # Logging
        log_dict = {
            f'{mode}_loss': total_loss,
            f'{mode}_acc': acc,
            f'{mode}_l_cls': l_bce,
            f'{mode}_l_spatial': l_spatial,
            f'{mode}_l_freq': l_freq,
            f'{mode}_l_joint': l_joint,
            f'{mode}_l_consistency': l_consistency,
            f'{mode}_lambda4': lambda4
        }
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
        
        return total_loss

    def training_step(self, batch, batch_idx):
        return self.calculate_losses(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_losses(batch, batch_idx, mode="val")
        
        # W&B Image Logging (Log occasionally, e.g., first batch of an epoch)
        if batch_idx == 0 and isinstance(self.logger, pl.loggers.WandbLogger):
            # Extract inputs and predictions for the first sample in the batch
            img_spatial = batch['img_spatial']
            img_freq = batch['img_freq']
            
            # Forward pass to generate decoders for visualization
            outputs = self(img_spatial, img_freq, run_decoders_override=True)
            
            # Convert tensors to loggable format
            spatial_pred = torch.sigmoid(outputs['spatial_mask'][0])
            freq_pred = torch.sigmoid(outputs['freq_mask'][0])
            
            # Gather Ground Truth if available
            gt_spatial = batch['clue_mask_gt'][0] if 'clue_mask_gt' in batch else torch.zeros_like(spatial_pred)
            
            # Convert tensors to CPU and NumPy formats suitable for wandb.Image logging.
            # Wandb on Windows sometimes struggles with raw temporary file mapping if the 
            # PyTorch CUDA tensor hasn't been properly flushed off the GPU and standardized.
            # Wandb expects image tensor inputs as channels-last (H, W, C) for standard numpy or PIL.
            img_spatial_cpu = img_spatial[0].detach().cpu().permute(1, 2, 0).numpy()
            gt_spatial_cpu = gt_spatial.detach().cpu().permute(1, 2, 0).numpy()
            spatial_pred_cpu = spatial_pred.detach().cpu().permute(1, 2, 0).numpy()
            freq_pred_cpu = freq_pred.detach().cpu().permute(1, 2, 0).numpy()
            
            # Log to WandB
            self.logger.experiment.log({
                "val_images": [
                    wandb.Image(img_spatial_cpu, caption="Input (Spatial)"),
                    wandb.Image(gt_spatial_cpu, caption="Ground Truth Spatial Mask"),
                    wandb.Image(spatial_pred_cpu, caption="Predicted Spatial Mask"),
                    wandb.Image(freq_pred_cpu, caption="Predicted Freq Mask")
                ]
            })
            
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.calculate_losses(batch, batch_idx, mode="test")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        
        # Adding a Learning Rate Scheduler: ReduceLROnPlateau
        # This will automatically reduce the LR if the validation loss stops improving.
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5,    # Reduce LR by half...
                patience=self.scheduler_patience,    # ...if no improvement for N epochs
                min_lr=1e-6
            ),
            'monitor': 'val_loss', # Metric to monitor
            'interval': 'epoch',
            'frequency': 1
        }
        
        return [optimizer], [lr_scheduler]
