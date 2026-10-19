"""
Author: Prof. Dr. Metin Zontul
Date: October 2026
Project: XAI-VisionMamba-KAN-Ensemble
Description: 15-Run Stratified Cross-Validation execution script across 3 seeds.
             Includes full training loops, loss optimization, and automated
             checkpoint saving to the logs/ directory.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import local modules from src
from src.hybrid_network import HybridResNetMambaKAN
from src.data_loader import get_kfold_dataloaders

def train_15_run_cv(data_dir, num_epochs=20):
    seeds = [42, 1024, 2026]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    # Ensure log directories exist
    os.makedirs("./logs/checkpoints", exist_ok=True)
    os.makedirs("./logs/metrics", exist_ok=True)
    
    # 3 Seeds * 5 Folds = 15 Runs
    for run_idx, seed in enumerate(seeds):
        print(f"\n{'='*40}")
        print(f"--- Starting Cross-Validation for Seed {seed} ---")
        print(f"{'='*40}")
        
        # Enforce deterministic behavior
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Extract the 5 folds for the current seed
        dataloaders = get_kfold_dataloaders(data_dir=data_dir, batch_size=32, n_splits=5, seed=seed)
        
        for fold, (train_loader, val_loader) in enumerate(dataloaders):
            print(f"\n>>> Seed: {seed} | Fold: {fold + 1}/5")
            
            # Initialize the base learner architecture
            # Adjust num_classes based on your exact Marble dataset structure (e.g., 5 classes)
            model = HybridResNetMambaKAN(num_classes=5).to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            
            best_val_acc = 0.0
            
            # Epoch loop
            for epoch in range(num_epochs):
                # --- Training Phase ---
                model.train()
                train_loss, train_correct, total_samples = 0.0, 0, 0
                
                for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    train_correct += torch.sum(preds == labels.data)
                    total_samples += inputs.size(0)
                    
                epoch_train_loss = train_loss / total_samples
                epoch_train_acc = (train_correct.double() / total_samples) * 100
                
                # --- Validation Phase ---
                model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                
                with torch.no_grad():
                    for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item() * inputs.size(0)
                        _, preds = torch.max(outputs, 1)
                        val_correct += torch.sum(preds == labels.data)
                        val_total += inputs.size(0)
                        
                epoch_val_loss = val_loss / val_total
                epoch_val_acc = (val_correct.double() / val_total) * 100
                
                # Step learning rate scheduler
                scheduler.step()
                
                print(f"Epoch {epoch+1:02d} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% "
                      f"|| Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
                
                # Save best fold model
                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
                    checkpoint_path = f"./logs/checkpoints/hybrid_s{seed}_f{fold+1}_best.pth"
                    torch.save(model.state_dict(), checkpoint_path)
            
            print(f"Best Validation Accuracy for Fold {fold + 1}: {best_val_acc:.2f}%")

if __name__ == "__main__":
    # Ensure the code is executed from the project root directory:
    # Example: python src/train.py
    target_data_directory = "./dataset/Marble"
    
    if not os.path.exists(target_data_directory):
        print(f"Error: Dataset path '{target_data_directory}' not found.")
        print("Please ensure your data is located in 'dataset/Marble/<Class_Names>/'.")
    else:
        train_15_run_cv(data_dir=target_data_directory, num_epochs=20)