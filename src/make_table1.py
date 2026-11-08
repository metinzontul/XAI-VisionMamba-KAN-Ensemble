"""
Author: Prof. Dr. Metin Zontul
Project: XAI-VisionMamba-KAN-Ensemble
File: make_table1.py

Builds the single Table 1 (epoch-20 accuracy, mean +- sample SD (n-1), min-max over 15 seed-fold runs).

  * All models of the controlled protocol (master_train.py and secondary_train.py: identical splits, schedule, loss, optimiser,
    epoch-20 weights): computed from results/<model>_v2_seed_<s>_fold_<f>.json (status "ok" only; failed runs are counted and listed).
    Groups: reference models (ResNet-50, ViT-B/16 with linear / MLP / KAN head), Mamba-based models (Vim-S, ResNet-Mamba-MLP/KAN),
    control (2nd Vim-S initialisation). Accuracy AND macro-F1 are available because the epoch-20 predictions were stored.
  * OLD models of the earlier manuscript version are NOT in the table by default (different protocol; their "KAN" heads contain no
    spline basis). With --with_old they are appended as a clearly separated, non-comparable group, read from old_models_epoch20.csv.

Usage:  python make_table1.py --run_dir . [--with_old] [--old_csv old_models_epoch20.csv] [--out_dir results/table1]
Outputs: table1.csv, table1.md, table1.tex
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os

import numpy as np

SEEDS = [42, 1024, 2026]
FOLDS = [1, 2, 3, 4, 5]
NEW = [  # (file key, name in the table, role); the order is the order of the table
    ("resnet50", "ResNet-50 (linear head)", "reference"),
    ("vit_b16", "ViT-B/16 (linear head)", "reference"),
    ("vit_b16_mlp", "ViT-B/16 + MLP head", "reference"),
    ("vit_b16_kan", "ViT-B/16 + KAN head (B-spline)", "reference"),
    ("vision_mamba", "Vision Mamba (Vim-S)", "mamba"),
    ("resnet_mamba_mlp", "ResNet-50 + Mamba block + MLP", "mamba"),
    ("resnet_mamba_kan", "ResNet-50 + Mamba block + KAN (B-spline)", "mamba"),
    ("vision_mamba_ctrl", "Vision Mamba (Vim-S), 2nd initialisation (control)", "control"),
]
GROUP_TITLE = {
    "reference": "Reference models (ImageNet pre-trained CNN and Transformer)",
    "mamba": "Mamba-based models",
    "control": "Control: identical recipe, other random initialisation",
    "old": "Earlier manuscript version - different protocol, NOT comparable (\"KAN\" heads there have no spline basis)",
}
FOOTNOTES = [
    "Mean ± sample SD (n-1) over 15 runs (3 seeds × stratified 5-fold CV, image-level splits identical for all models); accuracy after epoch 20, no checkpoint selection.",
    "Same recipe for all rows: focal loss (gamma 2), AdamW (lr 1e-4, weight decay 1e-2), batch 32, 20 epochs, constant learning rate, fp32, gradient clipping 1.0.",
    "KAN heads are B-spline Kolmogorov-Arnold layers (G = 5, cubic, grid [-3, 3]). All three ViT-B/16 head variants are preceded by the same parameter-free LayerNorm.",
    "Vim-S and its control were trained on Ubuntu (WSL2) with the official CUDA selective-scan kernel; all other models on Windows (same PyTorch 2.5.1).",
]


def sd(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


def load_new(run_dir):
    rows, failed = {}, {}
    for key, _, _ in NEW:
        acc, f1, bad = [], [], []
        for s, f in itertools.product(SEEDS, FOLDS):
            p = os.path.join(run_dir, "results", f"{key}_v2_seed_{s}_fold_{f}.json")
            if not os.path.isfile(p):
                continue
            with open(p, encoding="utf-8") as fh:
                r = json.load(fh)
            if r.get("status") == "ok":
                acc.append(r["val_acc"]); f1.append(r["macro_f1"])
            else:
                bad.append((s, f))
        if acc or bad:
            rows[key] = (np.array(acc), np.array(f1))
            failed[key] = bad
    return rows, failed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=".")
    ap.add_argument("--with_old", action="store_true", help="append the earlier-version models as a separate, non-comparable group")
    ap.add_argument("--old_csv", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "old_models_epoch20.csv"))
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args(argv)
    out = a.out_dir or os.path.join(a.run_dir, "results", "table1")
    os.makedirs(out, exist_ok=True)

    table = []   # dict(model, group, n, acc, sd, lo, hi, f1, f1sd, note)
    new, failed = load_new(a.run_dir)
    for key, name, role in NEW:
        if key not in new:
            print(f"[!] no results for {key} - row omitted")
            continue
        acc, f1 = new[key]
        if len(acc) == 0:
            print(f"[!] {key}: all runs failed - row omitted")
            continue
        note = "" if not failed[key] else f"{len(failed[key])} failed run(s): {failed[key]}"
        table.append(dict(model=name, group=role, n=len(acc), acc=float(acc.mean()), sd=sd(acc), lo=float(acc.min()), hi=float(acc.max()),
                          f1=float(f1.mean()), f1sd=sd(f1), note=note))

    if a.with_old:
        with open(a.old_csv, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                table.append(dict(model=r["model"], group="old", n=int(r["n"]), acc=float(r["acc_mean"]), sd=float(r["acc_sd"]),
                                  lo=float(r["acc_min"]), hi=float(r["acc_max"]), f1=None, f1sd=None,
                                  note=r.get("note", "").replace("+-", "±")))

    def fmt(r):
        f1 = "-" if r["f1"] is None else f"{r['f1']:.2f} ± {r['f1sd']:.2f}"
        return [r["model"], r["n"], f"{r['acc']:.2f} ± {r['sd']:.2f}", f"{r['lo']:.2f}–{r['hi']:.2f}", f1]

    head = ["Model", "Runs", "Accuracy % (mean ± SD)", "Min–max", "Macro-F1 % (mean ± SD)"]
    with open(os.path.join(out, "table1.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(head + ["group", "note"])
        for r in table:
            w.writerow(fmt(r) + [r["group"], r["note"]])

    md = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"]
    last = None
    for r in table:
        if r["group"] != last:
            title = GROUP_TITLE[r["group"]]
            md.append(f"| **{title}** |  |  |  |  |")
            last = r["group"]
        md.append("| " + " | ".join(str(x) for x in fmt(r)) + " |")
    md += [""] + FOOTNOTES
    if any(r["group"] == "old" for r in table):
        md += ["Earlier-version rows: epoch-20 accuracy read from the old training logs (AMP fp16, no gradient clipping, random-angle rotation, other RNG handling; "
               "no stored predictions, hence no macro-F1). Not a controlled comparison with the rows above."]
    notes = [f"- {r['model']}: {r['note']}" for r in table if r["note"]]
    if notes:
        md += ["", "Notes:"] + notes
    open(os.path.join(out, "table1.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    tex = ["\\begin{tabular}{lrccc}", "\\toprule", "Model & Runs & Accuracy (\\%) & Min--max & Macro-F1 (\\%) \\\\", "\\midrule"]
    last = None
    for r in table:
        if r["group"] != last:
            if last is not None:
                tex.append("\\midrule")
            tex.append("\\multicolumn{5}{l}{\\textit{" + GROUP_TITLE[r["group"]].replace('\"', "'") + "}} \\\\")
        last = r["group"]
        m, n, acc, rng, f1 = fmt(r)
        pm = "$\\pm$"
        acc, f1, rng = acc.replace("±", pm), f1.replace("±", pm), rng.replace("–", "--")
        tex.append(m + " & " + str(n) + " & " + acc + " & " + rng + " & " + f1 + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, "table1.tex"), "w", encoding="utf-8").write("\n".join(tex) + "\n")

    print("\n".join(md))
    print(f"\nFiles written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
