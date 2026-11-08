"""
Author: Prof. Dr. Metin Zontul
Project: XAI-VisionMamba-KAN-Ensemble
File: resnet_mamba_mlp.py

Control architecture: ResNet-50 + bidirectional Mamba + MLP classification head.

The convolutional trunk and the Mamba block are the shared `ResNetMambaBackbone`
(see mamba_core.py). `ResNetMambaKAN` (resnet_mamba_kan.py) uses the identical
backbone, so the MLP-vs-KAN comparison changes ONLY the classification head.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from mamba_core import ResNetMambaBackbone, active_scan_backend, count_parameters


class ResNetMambaMLP(nn.Module):
    HEAD_HIDDEN = 256
    HEAD_DROPOUT = 0.3

    def __init__(self, num_classes: int = 6, d_model: int = 512, pretrained: bool = True):
        super().__init__()
        self.backbone = ResNetMambaBackbone(pretrained=pretrained, d_model=d_model)
        self.mlp_head = nn.Sequential(                                  # Linear -> SiLU -> Dropout -> Linear
            nn.Linear(d_model, self.HEAD_HIDDEN),
            nn.SiLU(),
            nn.Dropout(self.HEAD_DROPOUT),
            nn.Linear(self.HEAD_HIDDEN, num_classes),
        )
        self.pretrained = pretrained

    def forward(self, x):
        return self.mlp_head(self.backbone(x))

    def describe(self) -> Dict:
        total, trainable = count_parameters(self)
        head = sum(p.numel() for p in self.mlp_head.parameters())
        return {
            "model": "ResNet-50 + bidirectional Mamba block + MLP head",
            "cnn_trunk": "torchvision ResNet-50, " + ("ImageNet-1k V2 weights (resnet50-11ad3fa6.pth)" if self.pretrained else "random init"),
            "mamba": {"blocks": 1, "d_model": 512, "d_state": 16, "d_conv": 4, "expand": 2,
                      "dt_rank": self.backbone.block.mixer.dt_rank, "tokens": 49,
                      "projection": "token-wise Linear(2048 -> 512) before the Mamba block",
                      "type": "bidirectional (Vim bimamba v2), pre-RMSNorm + residual"},
            "head": f"Linear(512,{self.HEAD_HIDDEN}) - SiLU - Dropout({self.HEAD_DROPOUT}) - Linear({self.HEAD_HIDDEN},6)",
            "head_parameters": head,
            "scan_backend": active_scan_backend(torch.cuda.is_available()),
            "parameters_total": total, "parameters_trainable": trainable,
        }
