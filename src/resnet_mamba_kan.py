"""
Author: Prof. Dr. Metin Zontul
Date: October 2026
Project: XAI-VisionMamba-KAN-Ensemble
Description: Fully integrated Hybrid ResNet-Mamba-KAN architecture. 
             Resolves Reviewer Point 17 by implementing authentic Bidirectional 
             Mamba blocks (mamba_ssm) and a mathematically correct B-spline 
             KAN classification head (Cox-de Boor recursion).
"""

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("Lütfen gerçek Mamba blokları için 'mamba_ssm' ve 'causal-conv1d' kütüphanelerini kurun.")

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        self.register_buffer('grid', torch.linspace(-1, 1, grid_size + spline_order + 1))
        self.spline_weights = nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order))
        self.base_weights = nn.Parameter(torch.randn(out_features, in_features))
        self.base_activation = nn.SiLU()

    def _compute_b_splines(self, x):
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:-1]) & (x < self.grid[1:])).float()
        
        for k in range(1, self.spline_order + 1):
            left_term = (x - self.grid[:-k-1]) / (self.grid[k:-1] - self.grid[:-k-1] + 1e-8)
            right_term = (self.grid[k+1:] - x) / (self.grid[k+1:] - self.grid[1:-k] + 1e-8)
            bases = left_term * bases[..., :-1] + right_term * bases[..., 1:]
            
        return bases

    def forward(self, x):
        base_output = nn.functional.linear(self.base_activation(x), self.base_weights)
        splines = self._compute_b_splines(x) 
        spline_output = torch.einsum('bik,oik->bo', splines, self.spline_weights)
        return base_output + spline_output

class BidirectionalMambaBlock(nn.Module):
    def __init__(self, d_model):
        super(BidirectionalMambaBlock, self).__init__()
        self.forward_mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.backward_mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        
    def forward(self, x):
        out_fwd = self.forward_mamba(x)
        x_flipped = torch.flip(x, dims=[1])
        out_bwd = self.backward_mamba(x_flipped)
        out_bwd_flipped = torch.flip(out_bwd, dims=[1])
        return out_fwd + out_bwd_flipped

class ResNetMambaKAN(nn.Module):
    def __init__(self, num_classes=6, d_model=512):
        super(ResNetMambaKAN, self).__init__()
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.flatten = nn.Flatten(2)
        self.mamba_block = BidirectionalMambaBlock(d_model=d_model)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.kan_head = KANLayer(in_features=d_model, out_features=num_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x) 
        batch_size, channels, h, w = features.size()
        seq = features.view(batch_size, channels, h * w) 
        seq = seq.permute(0, 2, 1) 
        mamba_out = self.mamba_block(seq) 
        mamba_out = mamba_out.permute(0, 2, 1)
        pooled = self.adaptive_pool(mamba_out).squeeze(-1) 
        logits = self.kan_head(pooled) 
        return logits