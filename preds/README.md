# Per-run predictions

120 files `<model>_v2_seed_<seed>_fold_<fold>.npz` = 8 models x 3 seeds x 5 folds, epoch-20 weights only.
Keys: `probs` (softmax probabilities), `logits`, `labels`, `val_idx` (indices into the ImageFolder order, equal to
`splits/splits_seed_<seed>.json`), `class_names`, `model`, `seed`, `fold`, `epoch` (= 20).
Models: `resnet50`, `vit_b16`, `vit_b16_mlp`, `vit_b16_kan`, `vision_mamba` (Vim-S), `vision_mamba_ctrl` (second Vim-S),
`resnet_mamba_mlp`, `resnet_mamba_kan`. The ensembles (`src/ensemble_eval.py`) are computed from these files.
