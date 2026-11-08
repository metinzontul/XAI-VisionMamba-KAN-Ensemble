# Source code

All scripts are run from this folder (`cd src`). Most take `--run_dir` (the run directory that contains `preds/`,
`results/`, `weights/`, `logs/`) and `--data` (the image root, see `../data/README.md`). The `--data` default in the
training scripts (`../../dataset/Marble`) is a leftover of the old layout: always pass `--data` explicitly.

## Training

| Script | Purpose |
|---|---|
| `master_train.py` | Trains Vim-S, ResNet-Mamba-MLP, ResNet-Mamba-KAN and (with `--with_vim_control`) the second Vim-S: 3 seeds x 5 folds, 20 epochs, epoch-20 weights only. Writes the unedited log, weights, validation probabilities (`preds/*.npz`), per-run metrics (`results/*.json`) and the split files. Resumes interrupted runs. |
| `secondary_train.py` | Same protocol for the reference models: ResNet-50 (linear head) and ViT-B/16 with linear, MLP and KAN head. |

## Models

| File | Content |
|---|---|
| `vision_mamba_train.py` | Complete Vim-S (24 blocks, bidirectional Mamba, middle class token); loads the official hustvl/Vim checkpoint. |
| `mamba_core.py` | Shared Mamba mixer and selective scan. Uses the official CUDA kernel if `mamba-ssm` is installed (WSL2/Linux), otherwise a pure-PyTorch parallel scan (Windows/CPU). |
| `resnet_mamba_mlp.py`, `resnet_mamba_kan.py` | ResNet-50 + bidirectional Mamba hybrids; identical backbone, MLP or B-spline KAN head. |
| `check_cuda_backend.py` | Compares the CUDA kernel with the PyTorch scan on a full Vim-S forward/backward pass (needs Linux, CUDA, `mamba-ssm`). |

## Analysis and figures

| Script | Input | Output |
|---|---|---|
| `make_table1.py` | `results/<model>_v2_seed_<s>_fold_<f>.json` | single-model table (mean, SD, min-max over 15 runs; failed runs counted and listed). `--with_old` appends the earlier-version models from `old_models_epoch20.csv` as a separate, non-comparable group (Supplementary Table S2). |
| `ensemble_eval.py` | `preds/*.npz` | ensembles E1-E5 (unweighted soft voting), paired tests, McNemar, Holm correction: `ensemble_report.txt`, `ensemble_runs.csv`, `ensemble_summary.csv`, `ensemble_tests.csv`, `error_overlap.csv`, `confusion_matrices.json`. |
| `plot_confusion_matrices.py` | `preds/*.npz` | pooled confusion matrices (every image counted once per seed). |
| `plot_learning_curves.py` | `logs/MASTER_*.txt` | learning curves (mean and SD over 15 runs) and per-run curves, drawn from the raw per-epoch log lines. |
| `xai_integrated_gradients.py` | `preds/`, `weights/`, images | Integrated-Gradients maps, deletion/insertion curves, overlap statistics, randomisation check. Needs the trained weights and `captum`. |
| `figio.py` | - | helper that writes each figure panel as its own 600-dpi file. |

## Re-running the analysis from the stored predictions

The repository contains the validation probabilities of all 120 runs (`../preds/`) and the split files (`../splits/`),
so the ensemble statistics and confusion matrices can be recomputed without training or GPU. From `src/`, with the
repository root as run directory:

```bash
mkdir -p ../results && cp ../splits/*.json ../results/      # enables the split consistency check (optional)
python ensemble_eval.py            --run_dir ..
python plot_confusion_matrices.py  --run_dir ..
```

Not included in the repository (too large or machine-specific): the trained weights (`weights/`, needed by the
Integrated-Gradients script) and the per-run `results/*.json` files (needed by `make_table1.py`). Both are created by
`master_train.py` and `secondary_train.py`; the per-fold accuracies are also in the logs and can be recomputed from
`preds/` (accuracy = mean of `argmax(probs) == labels`).

## Windows and WSL2

Hybrids, ResNet-50 and ViT-B/16 were trained on Windows with the PyTorch scan backend; Vim-S was trained under WSL2
with the CUDA kernel (`mamba-ssm` 2.2.4). `logs/README.txt` lists which log covers which model. The equivalence of the
two backends is tested in `../tests/`.
