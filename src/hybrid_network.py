"""
Author: Metin Zontul
Date: October 2026
Description: Hybrid ResNet-Mamba-KAN architecture. Integrates a single ResNet-34 
             feature extractor, authentic bidirectional state-space models (Mamba), 
             and a B-spline KAN classification head for 6-class macro-texture categorization.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from kan_layer import KANLayer

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("Please install 'mamba_ssm' and 'causal-conv1d' for authentic Mamba blocks.")

class BidirectionalMambaBlock(nn.Module):
    def __init__(self, d_model):
        super(BidirectionalMambaBlock, self).__init__()
        # Forward and backward Mamba state-space models with O(L) complexity
        self.forward_mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.backward_mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        out_fwd = self.forward_mamba(x)
        
        # Backward sequence processing
        x_flipped = torch.flip(x, dims=[1])
        out_bwd = self.backward_mamba(x_flipped)
        out_bwd_flipped = torch.flip(out_bwd, dims=[1])
        
        return out_fwd + out_bwd_flipped

class ResNetMambaKAN(nn.Module):
    def __init__(self, num_classes=6, d_model=512):
        super(ResNetMambaKAN, self).__init__()
        
        # 1. ResNet-34 Feature Extractor (Backbone)
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # 2. Sequence Flattening
        self.flatten = nn.Flatten(2)
        self.proj = nn.Linear(49, d_model) # Project 7x7 spatial map (49) to d_model
        
        # 3. Bidirectional Mamba Block
        self.mamba_block = BidirectionalMambaBlock(d_model=d_model)
        
        # Sequence pooling for classification head mapping
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 4. Kolmogorov-Arnold Network (KAN) Head
        self.kan_head = KANLayer(in_features=d_model, out_features=num_classes)
        
        # 5. Softmax layer for probability vector generation
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Input shape: (batch_size, 3, 224, 224)
        features = self.feature_extractor(x) # Shape: (batch_size, 512, 7, 7)
        
        # Flatten and permute for sequence modeling
        batch_size, channels, h, w = features.size()
        seq = features.view(batch_size, channels, h * w) 
        seq = seq.permute(0, 2, 1) # Shape: (batch, 49, 512)
        
        # Mamba sequence processing
        mamba_out = self.mamba_block(seq) 
        
        # Global pooling
        mamba_out = mamba_out.permute(0, 2, 1)
        pooled = self.adaptive_pool(mamba_out).squeeze(-1) 
        
        # KAN classification and Softmax
        logits = self.kan_head(pooled) 
        probabilities = self.softmax(logits) # Shape: (batch, num_classes)
        
        return probabilities