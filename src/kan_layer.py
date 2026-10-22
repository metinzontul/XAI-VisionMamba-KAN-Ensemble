"""
Author: Metin Zontul
Date: October 2026
Description: Kolmogorov-Arnold Network (KAN) layer implementation with authentic 
             B-spline basis functions using Cox-de Boor recursion. Replaces the 
             placeholder zero-tensor implementation to accurately evaluate spline contributions.
"""

import torch
import torch.nn as nn

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # B-Spline parameters and grid initialization
        self.grid = nn.Parameter(torch.linspace(-1, 1, grid_size + spline_order + 1), requires_grad=False)
        self.spline_weights = nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order))
        self.base_weights = nn.Parameter(torch.randn(out_features, in_features))
        self.base_activation = nn.SiLU()

    def _compute_b_splines(self, x):
        """
        Computes the B-spline basis functions using Cox-de Boor recursion.
        x shape: (batch_size, in_features)
        Returns shape: (batch_size, in_features, grid_size + spline_order)
        """
        x = x.unsqueeze(-1) # (batch_size, in_features, 1)
        bases = ((x >= self.grid[:-1]) & (x < self.grid[1:])).float()
        
        for k in range(1, self.spline_order + 1):
            left_term = (x - self.grid[:-k-1]) / (self.grid[k:-1] - self.grid[:-k-1] + 1e-8)
            right_term = (self.grid[k+1:] - x) / (self.grid[k+1:] - self.grid[1:-k] + 1e-8)
            bases = left_term * bases[..., :-1] + right_term * bases[..., 1:]
            
        return bases

    def forward(self, x):
        # Base linear activation branch
        base_output = nn.functional.linear(self.base_activation(x), self.base_weights)
        
        # B-Spline computation branch
        splines = self._compute_b_splines(x) 
        
        # Multiply by learnable spline weights and sum
        spline_output = torch.einsum('bik,oik->bo', splines, self.spline_weights)
        
        return base_output + spline_output