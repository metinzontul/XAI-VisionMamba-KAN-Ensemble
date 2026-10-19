"""
Author: Prof. Dr. Metin Zontul
Date: October 2026
Project: XAI-VisionMamba-KAN-Ensemble
Description: B-Spline based Kolmogorov-Arnold Network (KAN) layers 
             used as the highly explainable classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLinear(nn.Module):
    """
    1D KAN Layer leveraging B-spline control points for localized explainability.
    Ref: Eq. 8 and Eq. 9 of the manuscript.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Base activation weight (w_b)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Spline control points (c_i)
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
        nn.init.uniform_(self.spline_weight, -0.1, 0.1)

    def forward(self, x):
        # Base activation: SiLU
        base_out = F.linear(F.silu(x), self.base_weight)
        
        # B-spline evaluation logic (abstracted for clarity)
        # This replaces the fixed linear activation mapping with learnable univariate edges
        spline_out = self._compute_b_splines(x) 
        
        return base_out + spline_out
        
    def _compute_b_splines(self, x):
        # B-Spline interpolation using self.spline_weight across the grid
        # Output shape: (batch_size, out_features)
        # Placeholder for full spline calculation
        return torch.zeros((x.size(0), self.out_features), device=x.device)