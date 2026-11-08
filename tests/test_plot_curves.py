"""Synthetic test of plot_learning_curves.py: fake master logs (3 models x 3 seeds x 5 folds x 20 epochs), single-panel export."""
import os, sys, subprocess, tempfile, shutil
import numpy as np
from PIL import Image
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
rng = np.random.default_rng(0)
tmp = tempfile.mkdtemp()
lines = []
for m in ["vision_mamba", "resnet_mamba_mlp", "resnet_mamba_kan"]:
    for s in (42, 1024, 2026):
        for f in range(1, 6):
            lines.append(f"STARTING: {m} | seed {s} | fold {f}/5")
            for e in range(1, 21):
                lines.append(f"Epoch [{e}/20] - Train Loss: {0.6*np.exp(-e/6):.4f}, Train Acc: {60+1.8*e:.2f}% | Val Loss: {0.3*np.exp(-e/9)+0.1:.4f}, Val Acc: {75+0.8*e+rng.normal(0,2):.2f}% [10s]")
            lines.append(f"-> Fold {f} Completed.")
open(f"{tmp}/log.txt", "w").write("\n".join(lines))
r = subprocess.run([sys.executable, f"{SRC}/plot_learning_curves.py", "--logs", f"{tmp}/log.txt", "--out_dir", f"{tmp}/fig", "--paper_dir", f"{tmp}/paper", "--dpi", "80"], capture_output=True, text=True)
print(r.stdout[-1800:], r.stderr[-1500:]); assert r.returncode == 0
for f in ["article/Fig2a_learning_curve_vim_s.png", "article/Fig2b_learning_curve_hybrid_mlp.png", "article/Fig2c_learning_curve_hybrid_kan.png",
          "article/Fig2a_learning_curve_vim_s.pdf", "supplementary/FigS1a_val_acc_every_run_vim_s.png", "supplementary/FigS1c_val_acc_every_run_hybrid_kan.png"]:
    assert os.path.isfile(f"{tmp}/paper/{f}"), f
assert round(Image.open(f"{tmp}/paper/article/Fig2a_learning_curve_vim_s.png").info["dpi"][0]) == 600
assert os.path.isfile(f"{tmp}/fig/Figure_learning_curves.png") and os.path.isfile(f"{tmp}/fig/learning_curves_report.txt")
shutil.copy(f"{tmp}/paper/article/Fig2c_learning_curve_hybrid_kan.png", "/tmp/curve_test.png")
shutil.rmtree(tmp); print("LEARNING CURVE TESTS PASSED")
