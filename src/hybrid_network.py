"""
Author: Prof. Dr. Metin Zontul
Date: October 2026
Project: XAI-VisionMamba-KAN-Ensemble
Description: Hybrid ResNet-Mamba-KAN architecture combining convolutional 
             spatial priors, global state-space context, and KAN flexibility.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from src.kan_layer import KANLinear
# Assuming mamba_ssm is installed for the MambaBlock
# from mamba_ssm import Mamba 

class HybridResNetMambaKAN(nn.Module):
    def __init__(self, num_classes=5, d_model=512):
        super(HybridResNetMambaKAN, self).__init__()
        
        # 1. Convolutional Spatial Priors
        resnet = models.resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) # Output: (B, 512, 7, 7)
        
        # 2. Global State-Space Modeling (Mamba)
        # Flatten spatial dims to sequence: 7x7 = 49 patches
        self.d_model = d_model
        # self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba = nn.Linear(d_model, d_model) # Placeholder for Mamba block
        
        # 3. Explainable KAN Classification Head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.kan_head = KANLinear(in_features=d_model, out_features=num_classes)

    def forward(self, x):
        # Extract features
        x = self.backbone(x) # (B, 512, 7, 7)
        B, C, H, W = x.shape
        
        # Reshape to sequence (B, L, D) for Mamba
        x = x.view(B, C, H*W).permute(0, 2, 1) # (B, 49, 512)
        
        # Apply Bidirectional SSM
        x = self.mamba(x) # (B, 49, 512)
        
        # Pool and Classify via KAN
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1) # (B, 512)
        logits = self.kan_head(x)
        
        return logits