"""
Author: Prof. Dr. Metin Zontul
Project: XAI-VisionMamba-KAN-Ensemble
File: resnet_mamba_kan.py

Hybrid ResNet-50 + bidirectional Mamba + Kolmogorov-Arnold Network (KAN) head.

The backbone is the SAME `ResNetMambaBackbone` that `ResNetMambaMLP` uses; only the
classification head differs (KAN: G = 5 grid intervals, cubic B-splines r = 3).

KAN edge function (Eq. 9 of the manuscript):
        phi(x) = w_b * SiLU(x) + sum_i c_i * B_i(x),     i = 1 .. G + r.

Differences to the previous version (all documented in the manuscript's Methods):
  * the head input is the LayerNorm-ed pooled feature (zero mean / unit variance per sample),
    and the spline grid covers [-3, 3] (= the range of such features) instead of [-1, 1],
    so that most inputs fall inside the spline domain instead of only the SiLU base path;
  * initialisation: w_b ~ Kaiming-uniform (like nn.Linear), spline coefficients ~ N(0, 0.02^2)
    (the old N(0, 1) initialisation produced very large initial logits).
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from mamba_core import ResNetMambaBackbone, active_scan_backend, count_parameters


class KANLayer(nn.Module):
    """B-spline KAN layer: out_o = sum_i [ w_b[o,i] * SiLU(x_i) + sum_k c[o,i,k] * B_k(x_i) ]."""

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, grid_range=(-3.0, 3.0)):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order = grid_size, spline_order

        gmin, gmax = grid_range
        h = (gmax - gmin) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + gmin
        self.register_buffer("grid", grid)                              # G + 2r + 1 knots

        self.base_weights = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weights = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        nn.init.kaiming_uniform_(self.base_weights, a=math.sqrt(5))
        nn.init.normal_(self.spline_weights, mean=0.0, std=0.02)
        self.base_activation = nn.SiLU()

    def b_splines(self, x):
        """Cox-de Boor recursion. x: (b, in) -> (b, in, G + r)."""
        x = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[: -(k + 1)]) / (grid[k:-1] - grid[: -(k + 1)])
            right = (grid[k + 1:] - x) / (grid[k + 1:] - grid[1:-k])
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases

    def forward(self, x):
        base = nn.functional.linear(self.base_activation(x), self.base_weights)
        spline = torch.einsum("bik,oik->bo", self.b_splines(x), self.spline_weights)
        return base + spline


class ResNetMambaKAN(nn.Module):
    KAN_GRID = 5
    KAN_ORDER = 3

    def __init__(self, num_classes: int = 6, d_model: int = 512, pretrained: bool = True):
        super().__init__()
        self.backbone = ResNetMambaBackbone(pretrained=pretrained, d_model=d_model)
        self.kan_head = KANLayer(d_model, num_classes, grid_size=self.KAN_GRID, spline_order=self.KAN_ORDER)
        self.pretrained = pretrained

    def forward(self, x):
        return self.kan_head(self.backbone(x))

    def describe(self) -> Dict:
        total, trainable = count_parameters(self)
        head = sum(p.numel() for p in self.kan_head.parameters())
        return {
            "model": "ResNet-50 + bidirectional Mamba block + KAN head",
            "cnn_trunk": "torchvision ResNet-50, " + ("ImageNet-1k V2 weights (resnet50-11ad3fa6.pth)" if self.pretrained else "random init"),
            "mamba": {"blocks": 1, "d_model": 512, "d_state": 16, "d_conv": 4, "expand": 2,
                      "dt_rank": self.backbone.block.mixer.dt_rank, "tokens": 49,
                      "projection": "token-wise Linear(2048 -> 512) before the Mamba block",
                      "type": "bidirectional (Vim bimamba v2), pre-RMSNorm + residual"},
            "head": f"KAN 512 -> 6 (single layer), G={self.KAN_GRID}, r={self.KAN_ORDER}, grid range [-3, 3], "
                    f"{self.KAN_GRID + self.KAN_ORDER} B-spline bases + SiLU base path per edge",
            "head_parameters": head,
            "scan_backend": active_scan_backend(torch.cuda.is_available()),
            "parameters_total": total, "parameters_trainable": trainable,
        }
