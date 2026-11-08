"""Toy test of xai_integrated_gradients.py: random-weight models (2-block Vim), random images, real file formats."""
import json, os, sys, tempfile, shutil, time
import numpy as np, torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)
import vision_mamba_train as vmt
vmt.VIM_VARIANTS["small"] = (384, 2, "none")            # depth 2 instead of 24 (the 24-layer net was verified against the official code)
import xai_integrated_gradients as xai
import ensemble_eval as ev

rng = np.random.default_rng(0)
classes = ["Medium1", "Medium2", "Walnut1", "Walnut2", "White1", "White2"]
tmp = tempfile.mkdtemp()
data = f"{tmp}/Marble"
paths, y = [], []
for ci, c in enumerate(classes):
    os.makedirs(f"{data}/{c}")
    for k in range(10):
        base = rng.integers(0, 255, 3)
        arr = np.clip(base + rng.normal(0, 30, (64, 64, 3)), 0, 255).astype(np.uint8)
        p = f"{data}/{c}/img_{k:02d}.png"; Image.fromarray(arr).save(p); paths.append(p); y.append(ci)
y = np.array(y)
for d in ("weights", "preds", "results"):
    os.makedirs(f"{tmp}/{d}")
splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
json.dump({"seed": 42, "val_idx": [v.tolist() for _, v in splits]}, open(f"{tmp}/results/splits_seed_42.json", "w"))
t0 = time.time()
for f, (_, va) in enumerate(splits, 1):
    va = np.sort(va)
    for kind, fname in xai.FILES.items():
        model = xai.build_model(kind, 6, None, seed=100 * f + {"vim": 7, "kan": 3, "mlp": 5}[kind]).eval()
        torch.save(model.state_dict(), f"{tmp}/weights/{fname}_v2_seed_42_fold_{f}_epoch20.pth")
        with torch.no_grad():
            P = torch.cat([torch.softmax(model(xai.load_image(paths[i])), 1) for i in va]).numpy()
        np.savez_compressed(f"{tmp}/preds/{fname}_v2_seed_42_fold_{f}.npz", probs=P, logits=np.log(P), labels=y[va], val_idx=va,
                            class_names=np.array(classes), model=fname, seed=42, fold=f, epoch=20)
print("setup", round(time.time() - t0), "s")
out = f"{tmp}/out"
rc = xai.main(["--run_dir", tmp, "--data", data, "--out_dir", out, "--models", "vim", "kan", "mlp", "--n_per_class", "1", "--max_misclassified", "2", "--ig_steps", "4",
               "--ig_batch", "4", "--curve_steps", "5", "--n_random_check", "2", "--n_fig_rows", "2", "--per_image", "--dpi", "80", "--device", "cpu"])
assert rc == 0
shutil.copytree(f"{tmp}/results/paper_figures", f"{tmp}/pf_first")
shutil.rmtree(f"{tmp}/results/paper_figures")
assert os.path.isfile(f"{out}/xai_cache.pkl")
rc = xai.main(["--run_dir", tmp, "--data", data, "--out_dir", out, "--models", "vim", "kan", "mlp", "--n_per_class", "1", "--max_misclassified", "2",
               "--n_fig_rows", "2", "--dpi", "80", "--device", "cpu", "--from_cache"])
assert rc == 0
a1 = sorted(os.listdir(f"{tmp}/pf_first/article")); a2 = sorted(os.listdir(f"{tmp}/results/paper_figures/article"))
assert a1 == a2, (a1, a2)                                            # the re-drawn files are the same set
pf = f"{tmp}/results/paper_figures"
from PIL import Image as _I
for f in ["article/Fig4a_ig_main_original.png", "article/Fig4b_ig_main_vim_s.png", "article/Fig4c_ig_main_hybrid_mlp.png", "article/Fig4d_ig_main_hybrid_kan.png",
          "article/Fig5a_ig_overlap_iou.png", "article/Fig5b_ig_insertion_gain.png", "article/Fig5c_ig_deletion_gain.png", "article/Fig5d_ig_randomisation_check.png",
          "article/Fig5d_ig_randomisation_check.pdf", "supplementary/FigS2a_ig_classes_original.png", "supplementary/FigS2d_ig_classes_hybrid_kan.png",
          "supplementary/FigS3a_ig_misclassified_original.png", "supplementary/FigS3d_ig_misclassified_hybrid_kan.png"]:
    assert os.path.isfile(f"{pf}/{f}"), f
sizes = {_I.open(f"{pf}/article/Fig4{c}_ig_main_{n}.png").size for c, n in zip("abcd", ["original", "vim_s", "hybrid_mlp", "hybrid_kan"])}
assert len(sizes) == 1, sizes                                    # all columns exactly the same size
im = _I.open(f"{pf}/article/Fig4a_ig_main_original.png"); print("column size", im.size, "dpi", im.info.get("dpi"))
assert round(im.info["dpi"][0]) == 600
for f in ["Figure_IG_classes.png", "Figure_IG_misclassified.png", "Figure_IG_metrics.png", "Figure_IG_main.png", "xai_selection.csv", "xai_metrics.csv", "xai_summary.txt", "xai_maps_pooled.npz"]:
    assert os.path.isfile(f"{out}/{f}"), f
txt = open(f"{out}/xai_summary.txt").read()
import re
for m in re.finditer(r"(Vision Mamba \(Vim-S\)|ResNet-Mamba-KAN|ResNet-Mamba-MLP): (\d+)/(\d+)\s*(<--)?", txt):
    assert m.group(2) == m.group(3) and not m.group(4), m.group(0)       # stored predictions reproduced for every image
assert "MISMATCH" not in txt
for key in ("IG completeness", "Hybrid-MLP vs Hybrid-KAN", "Hybrid-MLP minus Hybrid-KAN", "outer two cell", "images reproduced by every model"):
    assert key in txt, key
assert abs(xai.chance_iou() - 0.112) < 0.01
# pooling and measures
m = np.arange(224 * 224, dtype=float).reshape(224, 224)
p = xai.pool(m); assert p.shape == (28, 28) and abs(p[0, 0] - m[:8, :8].mean()) < 1e-9
a = np.zeros((28, 28)); a[:5, :] = 1
assert xai.iou_top(a, a) == 1.0 and abs(xai.spearman(np.random.default_rng(1).random((28, 28)), np.random.default_rng(1).random((28, 28))) - 1) < 1e-9
# IG vs an independent Riemann estimate on a linear toy model: attribution = w * x (zero baseline)
lin = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 224 * 224, 6)); 
torch.manual_seed(0); x = torch.randn(1, 3, 224, 224)
mp, delta = xai.ig_positive_map(lin, x, 2, 8, 4)
exact = (lin[1].weight[2].view(1, 3, 224, 224) * x).sum(1)[0].clamp(min=0).detach().numpy()
assert np.allclose(mp, exact, atol=1e-4), np.abs(mp - exact).max()
# deletion curve of the true relevance order must not exceed the random order for a model that only looks at one corner
class Corner(torch.nn.Module):
    def forward(self, x):
        s = x[:, :, :56, :56].mean((1, 2, 3)); return torch.stack([-s, s], 1)
xc = torch.ones(1, 3, 224, 224)
rel = np.zeros((224, 224)); rel[:56, :56] = 1
cur, _ = xai.curves(Corner(), xc, xai.pool(rel), 1, 10)
assert cur["del_rel"] < cur["del_rand"] and cur["ins_rel"] > cur["ins_rand"], cur
print(open(f"{out}/xai_summary.txt").read())
shutil.copy(f"{out}/Figure_IG_classes.png", "/tmp/xai_test.png")
shutil.rmtree(tmp)
print("XAI SCRIPT TESTS PASSED", round(time.time() - t0), "s")
