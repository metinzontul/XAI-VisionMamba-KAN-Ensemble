"""Synthetic test of ensemble_eval.py: file format of master_train.py, independent re-computation of the tests."""
import json, os, subprocess, sys, tempfile, shutil
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)
import ensemble_eval as ev

rng = np.random.default_rng(0)
N, C = 996, 6
counts = [176, 170, 199, 137, 227, 87]
y_all = np.repeat(np.arange(C), counts)
classes = ["Medium1", "Medium2", "Walnut1", "Walnut2", "White1", "White2"]
tmp = tempfile.mkdtemp()
os.makedirs(f"{tmp}/preds"); os.makedirs(f"{tmp}/results")

def fake_probs(y, acc, shared_noise):
    """probabilities whose arg-max is right with prob ~acc; errors partly shared via shared_noise"""
    n = len(y)
    wrong = (rng.random(n) > acc) | (shared_noise & (rng.random(n) < 0.5))
    pred = np.where(wrong, (y + rng.integers(1, C, n)) % C, y)
    p = rng.dirichlet(np.ones(C) * 0.5, n) * 0.4
    p[np.arange(n), pred] += 0.6
    return p / p.sum(1, keepdims=True)

ACC = {"vision_mamba": 0.90, "resnet_mamba_kan": 0.88, "resnet_mamba_mlp": 0.875, "vision_mamba_ctrl": 0.90,
       "resnet50": 0.85, "vit_b16": 0.88, "vit_b16_mlp": 0.89, "vit_b16_kan": 0.895}
for s in ev.SEEDS:
    splits = list(StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(N), y_all))
    json.dump({"seed": s, "val_idx": [v.tolist() for _, v in splits]}, open(f"{tmp}/results/splits_seed_{s}.json", "w"))
    for f, (_, va) in enumerate(splits, 1):
        va = np.sort(va); y = y_all[va]
        hard = rng.random(len(va)) < 0.08
        for name, a in ACC.items():
            if name == "vision_mamba_ctrl" and s == 2026:      # missing runs -> E3 must be skipped for that seed? (all missing below)
                pass
            P = fake_probs(y, a, hard)
            np.savez_compressed(f"{tmp}/preds/{name}_v2_seed_{s}_fold_{f}.npz", probs=P.astype(np.float32), logits=np.log(P).astype(np.float32),
                                labels=y, val_idx=va, class_names=np.array(classes), model=name, seed=s, fold=f, epoch=20)

out = f"{tmp}/out"
r = subprocess.run([sys.executable, os.path.join(SRC, "ensemble_eval.py"), "--run_dir", tmp, "--out_dir", out], capture_output=True, text=True)
print(r.stdout[-6000:]); print(r.stderr[-2000:])
assert r.returncode == 0
for f in ["ensemble_report.txt", "ensemble_runs.csv", "ensemble_summary.csv", "ensemble_tests.csv", "confusion_matrices.json", "error_overlap.csv"]:
    assert os.path.isfile(f"{out}/{f}"), f

# ---- independent re-computation of E1 vs Vim ----
runs, cl = ev.load_runs(tmp, ev.SEEDS, ev.FOLDS)
d, nval, ntrain = [], [], []
for s in ev.SEEDS:
    for f in ev.FOLDS:
        v, k = runs["vim"][(s, f)], runs["kan"][(s, f)]
        ens = (v["probs"] + k["probs"]) / 2
        y = v["labels"]
        d.append(((ens.argmax(1) == y).mean() - (v["probs"].argmax(1) == y).mean()) * 100)
        nval.append(len(y)); ntrain.append(N - len(y))
d = np.array(d); J = len(d)
t = d.mean() / np.sqrt((1/J + np.mean(nval)/np.mean(ntrain)) * d.var(ddof=1))
p = 2 * stats.t.sf(abs(t), J - 1)
import csv
rows = list(csv.reader(open(f"{out}/ensemble_tests.csv", newline="")))
hdr = rows[0]; prim = [x for x in rows[1:] if x[0] == "primary"][0]
print("independent: mean diff %.4f t %.4f p %.6f | script: %s %s %s" % (d.mean(), t, p, prim[5], prim[8], prim[9]))
assert abs(float(prim[5]) - d.mean()) < 1e-3 and abs(float(prim[8]) - t) < 1e-3 and abs(float(prim[9]) - p) < 1e-5

# McNemar seed 42 independent
idx, ys, ps_e, ps_v = [], [], [], []
for f in ev.FOLDS:
    v, k = runs["vim"][(42, f)], runs["kan"][(42, f)]
    idx.append(v["val_idx"]); ys.append(v["labels"])
    ps_e.append(((v["probs"] + k["probs"]) / 2).argmax(1)); ps_v.append(v["probs"].argmax(1))
y = np.concatenate(ys); e = np.concatenate(ps_e) == y; vv = np.concatenate(ps_v) == y
b, c = int((e & ~vv).sum()), int((~e & vv).sum())
pm = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
print("independent McNemar seed 42:", b, c, pm, "| script:", prim[11:])
assert prim[11].startswith(f"{b},{c},") and abs(float(prim[11].split(",")[2]) - pm) < 1e-6

# D3 reference-model family: ViT KAN vs MLP independently recomputed
dd = []
runs_all, _ = ev.load_runs(tmp, ev.SEEDS, ev.FOLDS)
for s_ in ev.SEEDS:
    for f in ev.FOLDS:
        a_, b_ = runs_all["vitkan"][(s_, f)], runs_all["vitmlp"][(s_, f)]
        y_ = a_["labels"]
        dd.append(((a_["probs"].argmax(1) == y_).mean() - (b_["probs"].argmax(1) == y_).mean()) * 100)
dd = np.array(dd)
t3 = dd.mean() / np.sqrt((1/len(dd) + np.mean(nval)/np.mean(ntrain)) * dd.var(ddof=1))
p3 = 2 * stats.t.sf(abs(t3), len(dd) - 1)
ref_rows = [x for x in rows[1:] if x[0] == "reference_models"]
assert len(ref_rows) == 10, len(ref_rows)
kv = [x for x in ref_rows if x[2] == "vitkan" and x[3] == "vitmlp"][0]
print("independent ViT KAN-MLP: diff %.4f t %.4f p %.6f | script: %s %s %s" % (dd.mean(), t3, p3, kv[5], kv[8], kv[9]))
assert abs(float(kv[5]) - dd.mean()) < 1e-3 and abs(float(kv[9]) - p3) < 1e-5
# Holm within the D3 family: adjusted p >= raw p, monotone rule
assert all(float(x[10]) >= float(x[9]) - 1e-12 for x in ref_rows)
# the plan's tests are untouched by the new family: primary + 6 secondary + 4 base + 10 reference
assert sum(1 for x in rows[1:] if x[0] == "secondary") == 6 and sum(1 for x in rows[1:] if x[0] == "base_models") == 4
report = open(f"{out}/ensemble_report.txt", encoding="utf-8").read()
assert "D3. REFERENCE-MODEL COMPARISONS" in report and "KAN vs MLP head on ViT-B/16" in report
# ensembles contain only the plan's members
assert all(e in report for e in ("E1:", "E2:", "E3:", "E4:", "E5:")) and "ViT-B/16" not in [l for l in report.splitlines() if l.startswith("E5")][0]

# Holm
assert np.allclose(ev.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

# tie -> lowest class index; unweighted mean
r0 = dict(probs=np.array([[0.5, 0.5, 0, 0, 0, 0.0]]), labels=np.array([1]), val_idx=np.array([0]))
assert ev.predictions(r0)[0] == 0

# inconsistency detection
z = dict(np.load(f"{tmp}/preds/resnet_mamba_kan_v2_seed_42_fold_1.npz"))
z["val_idx"] = z["val_idx"][::-1].copy()
np.savez_compressed(f"{tmp}/preds/resnet_mamba_kan_v2_seed_42_fold_1.npz", **z)
r2 = subprocess.run([sys.executable, os.path.join(SRC, "ensemble_eval.py"), "--run_dir", tmp, "--out_dir", out + "2"], capture_output=True, text=True)
assert r2.returncode != 0 and "INCONSISTENT" in (r2.stdout + r2.stderr), r2.stdout[-500:] + r2.stderr[-500:]
print("inconsistent validation indices correctly stop the analysis")

# missing second Vim -> E3 skipped, report says so
for fn in os.listdir(f"{tmp}/preds"):
    if fn.startswith("vision_mamba_ctrl"): os.remove(f"{tmp}/preds/{fn}")
z["val_idx"] = z["val_idx"][::-1].copy()
np.savez_compressed(f"{tmp}/preds/resnet_mamba_kan_v2_seed_42_fold_1.npz", **z)
r3 = subprocess.run([sys.executable, os.path.join(SRC, "ensemble_eval.py"), "--run_dir", tmp, "--out_dir", out + "3"], capture_output=True, text=True)
assert r3.returncode == 0 and "E3 (two Vim-S) is NOT done" in r3.stdout, r3.stdout[-800:] + r3.stderr[-500:]
print("E3 skipped when the second Vim is missing")
shutil.rmtree(tmp)
print("ENSEMBLE EVAL TESTS PASSED")
