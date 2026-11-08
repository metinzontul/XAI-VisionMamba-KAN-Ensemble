Training logs of the experiments reported in the paper (unedited, as written by the programs).

All runs: 996 images, StratifiedKFold (5 folds) x seeds 42, 1024, 2026 = 15 runs per model, 20 epochs,
AdamW lr 1e-4, weight decay 0.01, batch 32, focal loss (gamma 2), final-epoch (epoch 20) weights only.
Environments: Windows 10 (torch selective-scan back-end) and Ubuntu on WSL2 (official CUDA selective-scan kernel,
mamba-ssm 2.2.4); PyTorch 2.5.1+cu121, torchvision 0.20.1, NVIDIA GeForce RTX 4080.

Training logs (all finish with "ALL TRAININGS COMPLETED")
  MASTER_TRAINING_LOG_20261105_171659.txt     Windows   ResNet-Mamba-MLP and ResNet-Mamba-KAN (2 x 15 runs)
  MASTER_TRAINING_LOG_20261105_204351.txt     WSL2      Vim-S (15 runs)
  MASTER_TRAINING_LOG_20261105_213908.txt     WSL2      Vim-S second training, "vision_mamba_ctrl" (15 runs)
  SECONDARY_TRAINING_LOG_20261106_181559.txt  Windows   ResNet-50, ViT-B/16 (linear, MLP and KAN heads) (4 x 15 runs)

Other logs (no model was trained for the paper in these)
  MASTER_TRAINING_LOG_20261105_170143.txt     Windows   speed benchmark only (timing estimate of the first three models)
  MASTER_TRAINING_LOG_20261105_202536.txt     WSL2      speed benchmark only (Vim-S with the CUDA kernel)
  SECONDARY_TRAINING_LOG_20261106_175121.txt  Windows   speed benchmark only (ResNet-50 and the three ViT-B/16 models)
  MASTER_TRAINING_LOG_20261105_203449.txt     WSL2      aborted at start: two checkpoint files in the Vim-small
                                                        repository; repeated successfully in ..._204351.txt

The logs contain the local user name and paths of the workstation; they were not edited.
Per-run predictions and the split files are in the code repository
(https://github.com/metinzontul/XAI-VisionMamba-KAN-Ensemble).
