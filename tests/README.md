# Tests

Nine self-checking scripts. None of them needs the marble images or any trained weights: they use synthetic data
(random images, fake probabilities, fake logs) and, where noted, shrunken models, so they run on a CPU in minutes.
Each script stops with an `AssertionError` on failure and prints a "PASSED"/"DONE" line on success. Run them from the
repository root, for example `python tests/test_kan_and_hybrid.py`.

| Script | Tests this code | What is checked | Needs |
|---|---|---|---|
| `test_scan_vs_official.py` | `src/mamba_core.py` | Our parallel scan equals the official sequential `selective_scan_ref` of Mamba/Vim (forward output and gradients); our Mamba mixer equals the official mixer maths. This is what allows the CPU/Windows scan backend to replace the CUDA kernel. | official reference code (`VIMSRC`) |
| `test_vim_vs_official.py` | `src/vision_mamba_train.py` (Vim-S) | The full 24-layer Vim-S has exactly the parameter names, shapes and parameter count of the official hustvl/Vim model (so the official checkpoint loads) and gives the same forward outputs. CUDA-only kernels are replaced by their official PyTorch reference equivalents on both sides. | official reference code (`VIMSRC`) |
| `test_kan_and_hybrid.py` | `src/resnet_mamba_kan.py`, `src/resnet_mamba_mlp.py` | B-spline KAN layer: knot and basis counts, partition of unity inside the grid range, at most r+1 non-zero bases, moderate initial logit scale, finite gradients. Both hybrids give `(batch, 6)` outputs, and the ResNet-Mamba backbone is the same in the MLP and KAN variants. | CPU |
| `test_pipeline_smoke.py` | `src/master_train.py` | End-to-end run on synthetic data (Vim shrunk to depth 2 for speed only): result JSONs and predictions are written, checkpoints are verified after saving, the validation indices and labels are identical across models, probabilities sum to 1, an interrupted run resumes without retraining and writes a new log, and a forced divergence is reported as FAILED instead of being counted silently. | CPU |
| `test_make_table1.py` | `src/make_table1.py` | All eight models appear in the correct groups; mean and standard deviation are recomputed independently; a failed run is counted, listed and excluded from the statistics; old rows appear only with `--with_old`. | CPU |
| `test_ensemble_eval.py` | `src/ensemble_eval.py` | The ensemble statistics (paired t-test, McNemar, Holm correction, reference-model comparisons) are recomputed independently and must match the script; all of E1-E5 appear in the report; inconsistent validation indices stop the analysis; E3 is skipped when the second Vim-S is missing. | CPU |
| `test_plot_confusion.py` | `src/plot_confusion_matrices.py` | The pooled confusion matrices contain every image three times (once per seed), the diagonal equals the number of correct predictions, macro-F1 equals scikit-learn's value, and the 600-dpi figure files are written. | CPU |
| `test_plot_curves.py` | `src/plot_learning_curves.py` | The learning-curve parser reads the log format of `master_train.py`; the article and supplementary figure files (Fig. 2a-c, Fig. S1a-c) are produced at 600 dpi. | CPU |
| `test_xai.py` | `src/xai_integrated_gradients.py` | Integrated-Gradients maps are cached and redrawn identically; the stored predictions are reproduced for every image (no MISMATCH); pooling, IoU and Spearman helpers behave as defined; the chance-level IoU is about 0.112; the patch-level attribution equals an exact computation; deletion and insertion curves for the relevance order beat random order. Uses random-weight models (2-block Vim) and random images. | CPU, `captum` |

## Official reference code

The two `*_vs_official.py` tests need a local copy of files from <https://github.com/hustvl/Vim> and
<https://github.com/state-spaces/mamba> (`vim_ssi.py`, `vim_mamba_simple.py`, `models_mamba.py`). Put them in one
folder and set `VIMSRC` (default `/tmp/vimsrc`):

```bash
VIMSRC=/path/to/reference python tests/test_scan_vs_official.py
VIMSRC=/path/to/reference python tests/test_vim_vs_official.py
```

## What the tests do not show

They verify the code, not the scientific results. The reported numbers come from the logs (`logs/`), the stored
predictions (`preds/`) and the analysis scripts run on them.
