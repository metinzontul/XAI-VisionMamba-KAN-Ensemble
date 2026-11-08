"""Synthetic test of plot_confusion_matrices.py (stored-prediction format of master_train.py)."""
import json, os, subprocess, sys, tempfile, shutil
import numpy as np
from sklearn.model_selection import StratifiedKFold
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)
import ensemble_eval as ev
import plot_confusion_matrices as pc

rng = np.random.default_rng(1)
N, C = 996, 6
y_all = np.repeat(np.arange(C), [176, 170, 199, 137, 227, 87])
classes = ["Medium1", "Medium2", "Walnut1", "Walnut2", "White1", "White2"]
tmp = tempfile.mkdtemp(); os.makedirs(f"{tmp}/preds"); os.makedirs(f"{tmp}/results")
ACC = {"vision_mamba": 0.90, "resnet_mamba_kan": 0.88, "resnet_mamba_mlp": 0.875, "vision_mamba_ctrl": 0.90}
def fake(y, acc):
    wrong = rng.random(len(y)) > acc
    pred = np.where(wrong, (y + rng.integers(1, C, len(y))) % C, y)
    p = rng.dirichlet(np.ones(C) * 0.5, len(y)) * 0.4; p[np.arange(len(y)), pred] += 0.6
    return p / p.sum(1, keepdims=True)
for s in ev.SEEDS:
    sp = list(StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(N), y_all))
    json.dump({"seed": s, "val_idx": [v.tolist() for _, v in sp]}, open(f"{tmp}/results/splits_seed_{s}.json", "w"))
    for f, (_, va) in enumerate(sp, 1):
        va = np.sort(va)
        for name, a in ACC.items():
            P = fake(y_all[va], a)
            np.savez_compressed(f"{tmp}/preds/{name}_v2_seed_{s}_fold_{f}.npz", probs=P.astype(np.float32), logits=np.log(P).astype(np.float32),
                                labels=y_all[va], val_idx=va, class_names=np.array(classes), model=name, seed=s, fold=f, epoch=20)
out = f"{tmp}/fig"
r = subprocess.run([sys.executable, os.path.join(SRC, "plot_confusion_matrices.py"), "--run_dir", tmp, "--out_dir", out, "--per_seed", "--dpi", "100",
                    "--systems", "vim", "kan", "E1", "E3"], capture_output=True, text=True)
print(r.stdout[-2500:]); print(r.stderr[-1500:])
assert r.returncode == 0
for f in ["Figure_confusion_matrices.png", "Figure_confusion_matrices.pdf", "Figure_confusion_matrix_seed_42.png",
          "confusion_matrices_aggregated.csv", "confusion_matrices_report.txt"]:
    assert os.path.isfile(f"{out}/{f}"), f
pf = f"{tmp}/results/paper_figures/article"
from PIL import Image
for f in ["Fig3a_confusion_vim_s.png", "Fig3b_confusion_hybrid_mlp.png", "Fig3c_confusion_hybrid_kan.png", "Fig3d_confusion_ensemble_E1.png", "Fig3a_confusion_vim_s.pdf"]:
    assert os.path.isfile(f"{pf}/{f}"), f
assert round(Image.open(f"{pf}/Fig3a_confusion_vim_s.png").info["dpi"][0]) == 600
# independent check: aggregated count total = 3 x 996, accuracy equals the mean of per-run accuracies (equal fold sizes up to 1 image)
runs, _ = ev.load_runs(tmp, ev.SEEDS, ev.FOLDS)
tot = 0; corr = 0
for k, rr in runs["vim"].items():
    tot += len(rr["labels"]); corr += (rr["probs"].argmax(1) == rr["labels"]).sum()
import csv
rows = list(csv.reader(open(f"{out}/confusion_matrices_aggregated.csv")))
cm = np.array([[int(x) for x in r_[2:]] for r_ in rows[1:7]])
assert cm.sum() == tot == 3 * N and np.trace(cm) == corr
# F1 from matrix equals sklearn on the concatenated predictions
from sklearn.metrics import f1_score
y = np.concatenate([rr["labels"] for rr in runs["vim"].values()]); p = np.concatenate([rr["probs"].argmax(1) for rr in runs["vim"].values()])
assert abs(pc.metrics_from_cm(cm)["macro_f1"] - f1_score(y, p, average="macro") * 100) < 1e-9
shutil.copy(f"{out}/Figure_confusion_matrices.png", "/tmp/cm_test.png")
shutil.rmtree(tmp)
print("CONFUSION MATRIX SCRIPT TESTS PASSED")
