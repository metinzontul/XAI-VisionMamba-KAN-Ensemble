# XAI-VisionMamba-KAN-Ensemble

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

**Official PyTorch implementation of the XAI-driven heterogeneous ensemble framework combining Vision Mamba (Vim) and Kolmogorov-Arnold Networks (KAN) for explainable macroscopic texture classification.**

This repository contains the source code, training logs, and pre-trained weights for the article focusing on the reliable and transparent classification of macroscopic marble textures. The proposed heterogeneous soft-voting framework achieves a peak Macro F1-Score of 97.17% by mathematically fusing the macroscopic stability of Vision Mamba with the hyper-flexible, localized explainability of B-Spline KAN heads.

## 📌 Project Overview
* **Base Learner 1 (Vision Mamba):** Provides global contextual understanding and highly stable predictive variance.
* **Base Learner 2 (Hybrid ResNet-Mamba-KAN):** Leverages learnable B-spline activation functions (Eq. 8 & 9) to precisely map high-frequency 2D spatial anomalies (micro-textures).
* **XAI Validation:** Utilizes Axiomatic Integrated Gradients (IG) to prove the superior spatial interpretability of the KAN classification head over standard MLPs (Figure 6).

## 📂 Repository Structure

```text
XAI-VisionMamba-KAN-Ensemble/
│
├── dataset/                    
│   └── Marble/                 # Place your dataset class folders here (e.g., Medium1, Wanlut1)
│
├── logs/                       
│   ├── ablation_study_logs/    # Detailed 15-run CV logs (Train/Val Acc & Loss)
│   ├── checkpoints/            # Pre-trained model weights (.pth) should be placed here
│   └── xai_heatmaps/           # Generated Integrated Gradients heatmaps will be saved here
│
├── src/                        
│   ├── data_loader.py          # Strict zero-leakage 5-fold CV & Z-score normalization
│   ├── kan_layer.py            # B-Spline KAN implementation
│   ├── hybrid_network.py       # ResNet-Mamba-KAN architecture
│   ├── ensemble.py             # Heterogeneous soft-voting fusion mechanism
│   ├── train.py                # 15-Run Stratified CV training execution
│   └── get_xai_maps.py         # Extracts and visualizes Integrated Gradients
│
└── requirements.txt            # Python dependencies
```

## 🚀 Installation & Setup

**1. Clone the repository:**
```bash
git clone https://github.com/metinzontul/XAI-VisionMamba-KAN-Ensemble.git
cd XAI-VisionMamba-KAN-Ensemble
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Download Pre-trained Champion Models (Weights):**
To evaluate the models or generate XAI heatmaps without training from scratch, download the pre-trained champion weights from the [Releases Page](https://github.com/metinzontul/XAI-VisionMamba-KAN-Ensemble/releases/tag/v1.0.0-weights).

Place the following downloaded files directly into the `logs/checkpoints/` directory:
*   `resnet_mamba_kan_seed_42_fold_3.pth` (Hybrid Model)
*   `vision_mamba_seed_1024_fold_1.pth` (Pure Vim Model)

## 💻 Usage

**Training the Models (15-Run Cross-Validation):**
Ensure your raw images are organized in `dataset/Marble/<Class_Name>/` and execute:
```bash
python src/train.py
```

**Generating XAI Heatmaps (Integrated Gradients):**
To replicate the interpretability analysis (Figure 6) using the downloaded champion weights:
```bash
python src/get_xai_maps.py
```
The resulting feature attribution overlays will be saved in `logs/xai_heatmaps/`.

## 📜 Citation

If you find this code or our conceptual framework useful in your research, please consider citing our paper:

```bibtex
@article{zontul2026xai,
  title={Structured Feature Compression in Vision Models: Comparing KAN, MLP, and Linear Classifier Heads for Macroscopic Texture Classification},
  author={Zontul, Metin and et al.},
  journal={MDPI Electronics},
  year={2026},
  publisher={MDPI}
}
```
**Author:** Prof. Dr. Metin Zontul (October 2026)