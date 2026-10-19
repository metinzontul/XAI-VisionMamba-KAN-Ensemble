\#\#\# Marble Texture Dataset Link

Due to GitHub's file size limitations for large image datasets, the original high-resolution marble texture images used for training the XAI-VisionMamba-KAN-Ensemble have been externally hosted.

You can access and download the complete \*\*Marble Dataset\*\* from the following Google Drive link:

🔗 \*\*\[Marble Dataset (Google Drive) \- Download Here\](https\://drive.google.com/drive/folders/1l3aJQqbNyKwCH3MylFaHuvH8cs\_Brc91?usp=sharing)\*\*

\#\#\#\# Dataset Structure  
Once downloaded, please ensure the dataset is extracted and placed directly inside this \`dataset/\` directory. The structure must match the PyTorch \`ImageFolder\` requirements, where each subfolder represents a distinct class:

\`\`\`text  
XAI-VisionMamba-KAN-Ensemble/  
│  
├── dataset/  
│   └── Marble/  
│       ├── Medium1/      \# Contains .jpg/.png images for Medium1 class  
│       ├── Medium2/  
│       ├── Wanlut1/  
│       ├── Wanlut2/  
│       ├── White1/  
│       └── White2/  
\`\`\`

After placing the images in the correct hierarchy, the \`src/data\_loader.py\` script will automatically load, resize (224x224), and apply Z-score normalization for the 15-Run Cross-Validation protocol.