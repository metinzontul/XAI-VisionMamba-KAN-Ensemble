"""
==============================================================================
Author: Prof. Dr. Metin Zontul & Assoc. Prof. Dr. Oguz Kaynar
Description: Integrated Gradients XAI Pipeline for Mamba & Hybrid Models.
             Pixel-level attribution analysis independent of spatial structures.
             Evaluates Champion models (Hybrid-KAN, Hybrid-MLP, Vision Mamba) 
             on sample marble textures for SCIE/Q1 ablation comparison.
==============================================================================
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import timm

from resnet_mamba_kan import ResNetMambaKAN

def load_vision_mamba(num_classes, weight_path, device):
    try:
        model = timm.create_model('mambaout_base', pretrained=False, num_classes=num_classes)
    except Exception:
        model = timm.create_model('vision_mamba_base', pretrained=False, num_classes=num_classes)
    model.to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    return model

def load_hybrid_kan_model(num_classes, weight_path, device):
    model = ResNetMambaKAN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    return model

def process_integrated_gradients(model, input_tensor, original_img, out_path, title_prefix, class_names):
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    pred_idx = output.argmax(dim=1).item()
    pred_class = class_names[pred_idx]
    prob = probs[0, pred_idx].item()
    
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(input_tensor, target=pred_idx, return_convergence_delta=True, n_steps=50)
    
    attributions_np = np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0))
    original_img_np = np.transpose(original_img.squeeze().cpu().detach().numpy(), (1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original_img_np = std * original_img_np + mean
    original_img_np = np.clip(original_img_np, 0, 1)

    fig, axis = viz.visualize_image_attr_multiple(
        attributions_np,
        original_img_np,
        methods=["original_image", "blended_heat_map"],
        signs=["all", "positive"],
        show_colorbar=True,
        titles=[f"Original Image", f"{title_prefix} IG\nPred: {pred_class} ({prob*100:.1f}%)"],
        fig_size=(10, 5),
        use_pyplot=False
    )
    
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"[*] IG plot saved: {out_path}")

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("[*] INITIALIZING INTEGRATED GRADIENTS (IG) XAI PIPELINE")
    print(f"{'='*80}")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 6
    CLASS_NAMES = ['Medium1', 'Medium2', 'Wanlut1', 'Wanlut2', 'White1', 'White2']
    
    HYBRID_KAN_WEIGHT_PATH = "weights/resnet_mamba_kan_seed_42_fold_3.pth"
    MAMBA_WEIGHT_PATH = "weights/mambavision_seed_1024_fold_1.pth" 
    
    DATA_DIR = "../dataset/Marble"
    OUTPUT_DIR = "plots/integrated_gradients"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    hybrid_kan_model = load_hybrid_kan_model(NUM_CLASSES, HYBRID_KAN_WEIGHT_PATH, DEVICE)
    mamba_model = load_vision_mamba(NUM_CLASSES, MAMBA_WEIGHT_PATH, DEVICE)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir): continue
            
        try:
             img_name = os.listdir(class_dir)[0]
             img_path = os.path.join(class_dir, img_name)
        except IndexError: continue
        
        pil_img = Image.open(img_path).convert('RGB')
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        input_tensor.requires_grad_()
        
        out_path_hybrid_kan = os.path.join(OUTPUT_DIR, f"IG_Hybrid_KAN_{class_name}.png")
        process_integrated_gradients(hybrid_kan_model, input_tensor, input_tensor, out_path_hybrid_kan, "Hybrid-KAN", CLASS_NAMES)
        
        out_path_mamba = os.path.join(OUTPUT_DIR, f"IG_Mamba_{class_name}.png")
        process_integrated_gradients(mamba_model, input_tensor, input_tensor, out_path_mamba, "Vision Mamba (Vim)", CLASS_NAMES)