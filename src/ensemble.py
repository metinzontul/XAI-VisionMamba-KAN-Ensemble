"""
Author: Prof. Dr. Metin Zontul
Date: October 2026
Project: XAI-VisionMamba-KAN-Ensemble
Description: Heterogeneous Soft-Voting Ensemble mathematically fusing 
             the macroscopic stability of Vision Mamba with the localized 
             explainability of the ResNet-Mamba-KAN architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HeterogeneousSoftVotingEnsemble(nn.Module):
    """
    Eq 10: y_hat = argmax (w1 * P_vim + w2 * P_hybrid)
    """
    def __init__(self, vim_model, hybrid_kan_model, w1=0.5, w2=0.5):
        super(HeterogeneousSoftVotingEnsemble, self).__init__()
        self.vim_model = vim_model
        self.hybrid_kan_model = hybrid_kan_model
        
        # Egalitarian weighting scheme
        self.w1 = w1
        self.w2 = w2
        
    def forward(self, x):
        # Get raw logits from both base learners
        logits_vim = self.vim_model(x)
        logits_hybrid = self.hybrid_kan_model(x)
        
        # Convert to probability distributions
        probs_vim = F.softmax(logits_vim, dim=1)
        probs_hybrid = F.softmax(logits_hybrid, dim=1)
        
        # Soft-voting fusion
        ensemble_probs = (self.w1 * probs_vim) + (self.w2 * probs_hybrid)
        
        return ensemble_probs