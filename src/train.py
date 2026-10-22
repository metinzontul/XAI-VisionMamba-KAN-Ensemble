"""
Author: Metin Zontul
Date: October 2026
Description: Training script for the Hybrid ResNet-Mamba-KAN model. 
             Implements strict seed and fold tracking to save corresponding 
             checkpoints for zero-leakage heterogeneous ensemble evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from hybrid_network import ResNetMambaKAN

# Experimental Hyperparameters matching the manuscript
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

def set_seed(seed):
    """Enforces reproducibility across PyTorch, NumPy, and Python runtime."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(seed, fold, train_loader, val_loader):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize hybrid model with 6 classes
    model = ResNetMambaKAN(num_classes=NUM_CLASSES).to(device)
    
    # Since the model outputs softmax probabilities, NLLLoss is used after log transformation.
    # Alternatively, remove softmax from hybrid_network.py and use CrossEntropyLoss directly.
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f"--- Training initialized | Seed: {seed} | Fold: {fold} | Device: {device} ---")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            probabilities = model(images)
            loss = criterion(torch.log(probabilities + 1e-8), labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Average Loss: {train_loss/len(train_loader):.4f}")
        
    # Save checkpoint mapped to specific seed and fold for ensemble pairing
    os.makedirs("checkpoints/hybrid", exist_ok=True)
    checkpoint_path = f"checkpoints/hybrid/model_seed{seed}_fold{fold}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[!] Model successfully saved to: {checkpoint_path}")

# Note: DataLoader instantiation and K-Fold cross-validation loops are executed here.