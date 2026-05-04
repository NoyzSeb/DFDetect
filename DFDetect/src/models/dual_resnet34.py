import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
import torch.nn.functional as F

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle varying image sizes (padding if dimensions don't match perfectly)
        if x.shape != skip.shape:
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNet34Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        # Extract layers for skip connections
        self.init_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 64 channels
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64
        self.layer2 = resnet.layer2  # 128
        self.layer3 = resnet.layer3  # 256
        self.layer4 = resnet.layer4  # 512

    def forward(self, x):
        features = {}
        features['c1'] = self.init_conv(x)
        x = self.maxpool(features['c1'])
        features['l1'] = self.layer1(x)
        features['l2'] = self.layer2(features['l1'])
        features['l3'] = self.layer3(features['l2'])
        features['l4'] = self.layer4(features['l3'])
        return features

class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec4 = UNetDecoderBlock(512, 256, 256)
        self.dec3 = UNetDecoderBlock(256, 128, 128)
        self.dec2 = UNetDecoderBlock(128, 64, 64)
        self.dec1 = UNetDecoderBlock(64, 64, 64)
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1) # 3 channels for RGB

    def forward(self, features):
        x = self.dec4(features['l4'], features['l3'])
        x = self.dec3(x, features['l2'])
        x = self.dec2(x, features['l1'])
        x = self.dec1(x, features['c1'])
        x = self.final_up(x)
        x = self.final_conv(x)
        return x

class DualResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoders
        self.spatial_encoder = ResNet34Encoder()
        self.freq_encoder = ResNet34Encoder()
        
        # Classification Head (Binary output)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 512 from spatial + 512 from freq = 1024
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1) # Logit output for Real vs Fake
        )
        
        # Decoders for Masking
        self.spatial_decoder = UNetDecoder()
        self.freq_decoder = UNetDecoder()

    def forward(self, img_spatial, img_freq, run_decoders_override=None):
        spatial_features = self.spatial_encoder(img_spatial)
        freq_features = self.freq_encoder(img_freq)
        
        # Classification
        spatial_pool = self.global_pool(spatial_features['l4']).flatten(1)
        freq_pool = self.global_pool(freq_features['l4']).flatten(1)
        
        concat_features = torch.cat([spatial_pool, freq_pool], dim=1)
        cls_logits = self.classifier(concat_features)
        
        outputs = {'cls_logits': cls_logits}
        
        # Determine whether to generate masks. 
        # In inference, we check if the model predicts 'Fake' (logit > 0 typically represents fake).
        # We also allow an override (run_decoders_override) to force it during training.
        generate_masks = run_decoders_override
        if generate_masks is None:
            # Inference mode: only decode for samples predicted as Fake
            # Using logit > 0 as standard threshold for binary cross-entropy (0.5 prob after sigmoid)
            generate_masks = (cls_logits.detach() > 0).any().item()
            
        if generate_masks:
            outputs['spatial_mask'] = self.spatial_decoder(spatial_features)
            outputs['freq_mask'] = self.freq_decoder(freq_features)
        else:
            outputs['spatial_mask'] = None
            outputs['freq_mask'] = None
            
        return outputs
