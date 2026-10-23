"""
Author: Prof. Dr. Metin Zontul
Date: October 2026
Project: XAI-VisionMamba-KAN-Ensemble
Description: Evaluates the Heterogeneous Soft-Voting Ensemble on the validation
             subset (Seed 2026, Fold 3) using actual dataset images. Computes the 
             authentic confusion matrix and exact Macro F1 score to resolve 
             reviewer discrepancies fully reproducibly without fabrication.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Import models dynamically to match the user's specific training files
import timm

# The hybrid class definition based exactly on the user's provided file
from resnet_mamba_kan_train_cv import HybridResNetMambaKAN

CLASS_NAMES = ['Medium1', 'Medium2', 'Wanlut1', 'Wanlut2', 'White1', 'White2']
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset path from previous screenshot validation
DATASET_PATH = r"C:\Users\ASUS-WORKSTATION\Desktop\metin\Oğuz\dataset\Marble"

def create_mambavision(num_classes):
    """Fallback logic used by the user's mambavision script."""
    try:
        model = timm.create_model('mambaout_base', pretrained=False, num_classes=num_classes)
    except Exception as e:
        model = timm.create_model('vision_mamba_base', pretrained=False, num_classes=num_classes)
    return model

def generate_authentic_ensemble_cm():
    print(f"--- Authentic Ensemble CM Generation on {DEVICE} ---")
    
    # 1. Initialize models and Load Weights
    try:
        # Load Vim based on user's exact fallback logic in mambavision_train_cv.py
        vim_model = create_mambavision(num_classes=NUM_CLASSES).to(DEVICE)
        
        # Load Hybrid model using exact class name found in resnet_mamba_kan_train_cv.py
        hybrid_model = HybridResNetMambaKAN(num_classes=NUM_CLASSES).to(DEVICE)
        
        vim_weight = "weights/vision_mamba_seed_2026_fold_3.pth"
        hybrid_weight = "weights/resnet_mamba_kan_seed_2026_fold_3.pth"
        
        vim_model.load_state_dict(torch.load(vim_weight, map_location=DEVICE))
        hybrid_model.load_state_dict(torch.load(hybrid_weight, map_location=DEVICE))
        
        vim_model.eval()
        hybrid_model.eval()
        print("[+] Models and weights (Seed 2026, Fold 3) loaded successfully.")
    except Exception as e:
        print(f"[-] ERROR: Model or checkpoint could not be loaded. Please check weight paths.\nDetails: {e}")
        return

    # 2. Load Dataset and Split for Seed 2026, Fold 3
    print("\n[*] Loading dataset and splitting for validation...")
    
    # Validation transform conforming to "ImageNet normalization" rule in the paper
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
        print(f"[+] Total images: {len(full_dataset)}")
    except Exception as e:
        print(f"[-] ERROR: Dataset not found. Check DATASET_PATH: {DATASET_PATH}\nDetails: {e}")
        return

    # 5-Fold Stratified split preserving class distribution (Random seed: 2026)
    targets = full_dataset.targets
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    
    # Get validation indices for Fold 3 (index 2)
    val_idx = None
    for fold, (train_index, test_index) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        if fold == 2:  # 0, 1, 2 -> 3rd Fold
            val_idx = test_index
            break
            
    val_subset = Subset(full_dataset, val_idx)
    # Using pin_memory=True and num_workers=4 based on the user's cv script
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    print(f"[+] Fold 3 Validation set isolated. Number of images: {len(val_subset)} (Expected: 199)")

    # 3. Execute Authentic Predictions
    all_preds = []
    all_labels = []
    
    print("\n[*] Computing Soft-Voting predictions...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            
            # Use torch.amp.autocast based on user's training loop implementation
            with torch.amp.autocast('cuda'):
                # Extract logit outputs
                logits_vim = vim_model(images)
                logits_hybrid = hybrid_model(images)
            
            # Convert to Softmax probabilities (Equation 10 in the paper)
            p_vim = torch.softmax(logits_vim, dim=1)
            p_hybrid = torch.softmax(logits_hybrid, dim=1)
            
            # Equal-weight Ensemble (w1=0.5, w2=0.5)
            p_ens = (0.5 * p_vim) + (0.5 * p_hybrid)
            y_hat = torch.argmax(p_ens, dim=1).cpu().numpy()
            
            all_preds.extend(y_hat)
            all_labels.extend(labels.numpy())

    # 4. Calculate Authentic (Non-Fabricated) Metrics
    acc = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    
    print("\n=======================================================")
    print("      VERIFIED S2026-F3 ENSEMBLE RESULTS")
    print("=======================================================")
    print(f"Evaluated Samples : {len(all_labels)}")
    print(f"Accuracy          : {acc:.4f}%")
    print(f"Macro F1-Score    : {macro_f1:.4f}%")
    print("=======================================================")
    print("[!] ATTENTION: Ensure Table 2 and the Abstract match this Macro F1 value exactly.")

    # 5. Visualization for Manuscript (Figure 3c)
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 12})
    
    plt.title("Soft-Voting Ensemble\n(Selected Evaluation: S2026_F3)", fontsize=14, pad=15, fontweight='bold')
    plt.xlabel("Predicted Class", fontsize=12, fontweight='bold')
    plt.ylabel("True Class", fontsize=12, fontweight='bold')
    
    # Save in high-resolution SCIE standard
    out_file = "Figure3c_Authentic_Ensemble_CM.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"\n[+] Authentic confusion matrix generated successfully: {out_file}")

if __name__ == "__main__":
    generate_authentic_ensemble_cm()