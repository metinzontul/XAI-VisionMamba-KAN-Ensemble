"""
Author: Prof. Dr. Metin Zontul
Date: October 2026
Project: XAI-VisionMamba-KAN-Ensemble
Description: Heterogeneous Soft-Voting Ensemble module and the 15-run 
             cross-validation evaluation script. Pairs Vim and Hybrid base 
             learners by identical seed-fold indices to compute variance metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import timm
from sklearn.metrics import accuracy_score, f1_score
from resnet_mamba_kan import ResNetMambaKAN

NUM_CLASSES = 6
SEEDS = [42, 1024, 2026]
FOLDS = 5

class HeterogeneousSoftVotingEnsemble(nn.Module):
    def __init__(self, vim_model, hybrid_kan_model, w1=0.5, w2=0.5):
        super(HeterogeneousSoftVotingEnsemble, self).__init__()
        self.vim_model = vim_model
        self.hybrid_kan_model = hybrid_kan_model
        self.w1 = w1
        self.w2 = w2
        
    def forward(self, x):
        # Obtain raw logits and convert to probabilities
        logits_vim = self.vim_model(x)
        logits_hybrid = self.hybrid_kan_model(x)
        
        probs_vim = F.softmax(logits_vim, dim=1)
        probs_hybrid = F.softmax(logits_hybrid, dim=1)
        
        # Soft-voting fusion (Equation 10)
        ensemble_probs = (self.w1 * probs_vim) + (self.w2 * probs_hybrid)
        return ensemble_probs

def evaluate_heterogeneous_ensemble(val_loaders):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Commencing 15-Run Heterogeneous Ensemble Evaluation on {device} ---")
    
    results = []
    
    for seed in SEEDS:
        for fold in range(1, FOLDS + 1):
            val_loader = val_loaders[f"{seed}_{fold}"]
            
            # Initialize base architectures
            vim_model = timm.create_model('vision_mamba_base', pretrained=False, num_classes=NUM_CLASSES).to(device)
            hybrid_model = ResNetMambaKAN(num_classes=NUM_CLASSES).to(device)
            
            vim_ckpt = f"checkpoints/vim/model_seed{seed}_fold{fold}.pth"
            hybrid_ckpt = f"checkpoints/hybrid/model_seed{seed}_fold{fold}.pth"
            
            vim_model.load_state_dict(torch.load(vim_ckpt, map_location=device))
            hybrid_model.load_state_dict(torch.load(hybrid_ckpt, map_location=device))
            
            vim_model.eval()
            hybrid_model.eval()
            
            ensemble_model = HeterogeneousSoftVotingEnsemble(vim_model, hybrid_model)
            ensemble_model.eval()
            
            all_preds, all_labels = [], []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    p_ens = ensemble_model(images)
                    y_hat = torch.argmax(p_ens, dim=1).cpu().numpy()
                    all_preds.extend(y_hat)
                    all_labels.extend(labels.numpy())
            
            acc = accuracy_score(all_labels, all_preds) * 100
            macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
            
            results.append({
                "Seed": seed,
                "Fold": fold,
                "Accuracy": acc,
                "Macro_F1": macro_f1
            })
            print(f"Evaluation Completed -> Seed: {seed}, Fold: {fold} | Accuracy: {acc:.2f}%, Macro F1: {macro_f1:.2f}%")

    df_results = pd.DataFrame(results)
    log_path = "15_run_ensemble_results.csv"
    df_results.to_csv(log_path, index=False)
    print(f"\n[!] Complete 15-run metrics saved to: {log_path}")
    
    mean_acc = df_results["Accuracy"].mean()
    std_acc = df_results["Accuracy"].std(ddof=1)
    
    print("\n=======================================================")
    print("      FINAL 15-RUN ENSEMBLE PERFORMANCE SUMMARY")
    print("=======================================================")
    print(f"Overall Accuracy (Mean ± SD) : {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Peak Macro F1-Score          : {df_results['Macro_F1'].max():.2f}%")
    print("=======================================================")