"""
Author: Prof. Dr. Metin Zontul
Date: October 2026
Project: XAI-VisionMamba-KAN-Ensemble
Description: Extracts and visualizes Axiomatic Integrated Gradients (IG) 
             feature attribution maps to evaluate the 2D spatial interpretability 
             of the KAN classification heads. Replicates Figure 6 of the manuscript.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

# Import the base learner from the project
from src.hybrid_network import HybridResNetMambaKAN

def load_image(image_path):
    """
    Loads an image and applies the validation transform (Z-score normalization).
    """
    # Eq 1: Normalization params
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0) # Add batch dimension
    
    # Keep original image for overlay plotting
    original_image = np.array(img.resize((224, 224)))
    return input_tensor, original_image

def generate_ig_heatmap(model, input_tensor, original_image, target_class, save_path):
    """
    Computes Integrated Gradients and saves the resulting heatmap overlay.
    """
    model.eval()
    
    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)
    
    # Compute attributions
    # Baseline is automatically zero (black image)
    attributions, delta = ig.attribute(input_tensor, target=target_class, return_convergence_delta=True)
    
    # Convert tensors to numpy arrays for Captum visualization
    attributions_np = np.transpose(attributions.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    
    # Create the visualization overlay
    fig, axis = viz.visualize_image_attr(
        attributions_np,
        original_image,
        method="blended_heat_map",
        sign="positive", # Show regions that positively contributed to the prediction
        show_colorbar=True,
        title=f"Integrated Gradients (Class {target_class})",
        use_pyplot=False
    )
    
    # Save the figure to the logs directory
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved IG heatmap to: {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure the output directory exists
    os.makedirs("./logs/xai_heatmaps", exist_ok=True)
    
    # Initialize model and load trained weights (Checkpoint)
    model = HybridResNetMambaKAN(num_classes=5).to(device)
    
    # NOTE: Ensure you have a trained checkpoint in the logs/checkpoints folder
    checkpoint_path = "./logs/checkpoints/hybrid_s1024_f1_best.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Model weights loaded successfully.")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using untrained weights for demonstration.")
    
    # Example Target: Medium1 Class Image
    sample_image_path = "./dataset/Marble/Medium1/sample_medium1.jpg"
    
    if os.path.exists(sample_image_path):
        input_tensor, original_image = load_image(sample_image_path)
        input_tensor = input_tensor.to(device)
        
        # Assume Target Class Index is 0 for Medium1
        target_class_idx = 0 
        
        save_file = "./logs/xai_heatmaps/IG_Hybrid_KAN_Medium1.png"
        generate_ig_heatmap(model, input_tensor, original_image, target_class_idx, save_file)
    else:
        print(f"Error: Sample image not found at {sample_image_path}. Please provide a valid marble texture image.")