"""Synthetic test of make_table1.py: all eight controlled models appear in the right groups; old rows only with --with_old."""
import json, os, subprocess, sys, tempfile, shutil, csv
import numpy as np
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
rng = np.random.default_rng(1)
tmp = tempfile.mkdtemp(); os.makedirs(f"{tmp}/results")
keys = ["resnet50", "vit_b16", "vit_b16_mlp", "vit_b16_kan", "vision_mamba", "resnet_mamba_mlp", "resnet_mamba_kan", "vision_mamba_ctrl"]
ACC = {}
for i, k in enumerate(keys):
    ACC[k] = []
    for s in (42, 1024, 2026):
        for f in range(1, 6):
            a = 80 + i + rng.normal(0, 2)
            ACC[k].append(a)
            st = "failed" if (k == "vit_b16" and s == 2026 and f == 5) else "ok"
            json.dump(dict(model=k, seed=s, fold=f, status=st, val_acc=a, macro_f1=a - 1) if st == "ok" else dict(model=k, status=st),
                      open(f"{tmp}/results/{k}_v2_seed_{s}_fold_{f}.json", "w"))
out = f"{tmp}/t1"
r = subprocess.run([sys.executable, os.path.join(SRC, "make_table1.py"), "--run_dir", tmp, "--out_dir", out], capture_output=True, text=True)
print(r.stdout[-3500:], r.stderr[-1500:]); assert r.returncode == 0
rows = list(csv.reader(open(f"{out}/table1.csv", newline="", encoding="utf-8")))[1:]
assert [x[0] for x in rows][:4] == ["ResNet-50 (linear head)", "ViT-B/16 (linear head)", "ViT-B/16 + MLP head", "ViT-B/16 + KAN head (B-spline)"]
assert [x[5] for x in rows] == ["reference"] * 4 + ["mamba"] * 3 + ["control"] and len(rows) == 8
vit = rows[1]; assert vit[1] == "14" and "failed run" in vit[6]            # failed run counted, listed, excluded
a = np.array(ACC["resnet50"]); assert vit and rows[0][2] == f"{a.mean():.2f} ± {a.std(ddof=1):.2f}"
md = open(f"{out}/table1.md", encoding="utf-8").read()
assert "Earlier manuscript version" not in md and "WSL2" in md
tex = open(f"{out}/table1.tex", encoding="utf-8").read()
assert tex.count("multicolumn") == 3 and "begin{tabular}{lrccc}" in tex
old = f"{tmp}/old.csv"; open(old, "w").write("model,n,acc_mean,acc_sd,acc_min,acc_max,note\nViT-MLP,15,86.55,3.15,81.41,90.45,\"a +- b\"\n")
r2 = subprocess.run([sys.executable, os.path.join(SRC, "make_table1.py"), "--run_dir", tmp, "--out_dir", out + "2", "--with_old", "--old_csv", old], capture_output=True, text=True)
assert r2.returncode == 0
md2 = open(f"{out}2/table1.md", encoding="utf-8").read()
assert "Earlier manuscript version" in md2 and "a ± b" in md2 and "NOT comparable" in md2
shutil.rmtree(tmp); print("MAKE_TABLE1 TESTS PASSED")
