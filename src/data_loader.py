"""
Author: Prof. Dr. Metin Zontul
Date: October 2026
Project: XAI-VisionMamba-KAN-Ensemble
Description: Data loader module implementing ImageNet Z-score normalization 
             and strict zero-leakage 5-fold cross-validation for macroscopic 
             marble textures.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np

class DatasetWrapper(Dataset):
    """
    Custom wrapper to ensure zero-data leakage by applying 
    data augmentations strictly to the training fold subsets.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def get_transforms():
    """
    Returns the training and validation transformations.
    Eq 1: ImageNet statistical priors for Z-score normalization.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def get_kfold_dataloaders(data_dir, batch_size=32, n_splits=5, seed=42):
    """
    Generates K-Fold DataLoaders. 
    Returns a list of tuples: [(train_loader_1, val_loader_1), ...]
    """
    train_tf, val_tf = get_transforms()
    
    # Load raw dataset without transforms (handled by wrapper)
    # Converts PIL Images to base structure
    dataset = datasets.ImageFolder(root=data_dir)
    labels = dataset.targets
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    dataloaders = []
    # Splitting based on strict stratification
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        
        # Create PyTorch subsets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        # Wrap subsets to apply isolated transforms
        train_data = DatasetWrapper(train_subset, transform=train_tf)
        val_data = DatasetWrapper(val_subset, transform=val_tf)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        dataloaders.append((train_loader, val_loader))
        
    return dataloaders