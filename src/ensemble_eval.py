"""
Author: Prof. Dr. Metin Zontul
Project: XAI-VisionMamba-KAN-Ensemble
File: ensemble_eval.py

Ensemble analysis exactly as fixed in the "Ensemble Analysis Plan" (written before any ensemble result existed).

Input : preds/<model>_v2_seed_<s>_fold_<f>.npz   (written by master_train.py; epoch-20 probabilities)
        results/splits_seed_<s>.json              (optional: validation indices, used for a consistency check)
Output: <out_dir>/ensemble_report.txt  (everything that is printed, unedited)
        ensemble_runs.csv       one row per (system, seed, fold): accuracy, macro-F1, per-class recall
        ensemble_summary.csv    mean / SD (n-1) / min / max over runs
        ensemble_tests.csv      all statistical tests
        confusion_matrices.json pooled confusion matrix per system and seed
        error_overlap.csv       pairwise error overlap of the base models

Rules implemented (plan sections 3-7):
  * Ensemble probability = UNWEIGHTED mean of the members' softmax probabilities; arg-max, ties -> lowest class index.
    Nothing is fitted on the validation data.
  * An ensemble is computed only for (seed, fold) pairs where ALL its members exist; the number of runs is printed.
  * Validation indices and labels must be identical across models for every (seed, fold) - otherwise the script stops.
  * Primary test: E1 versus Vision Mamba alone, Nadeau-Bengio corrected paired t-test on accuracy (not multiplicity-corrected).
  * McNemar (exact binomial) per seed on the seed's pooled predictions (never pooled across seeds).
  * Secondary family (Holm-corrected): E2..E5 each vs its best member, E1 vs E2, E1 vs E3.
  * E1 is called an improvement only if it beats Vision Mamba in the primary test (mean difference > 0, p < 0.05).
  * Base-model comparisons (section D2; NOT in the original plan, added after the results at the request of reviewer 2, therefore
    exploratory; own Holm family of four, so the p-values of the plan's six secondary tests are not affected):
    KAN vs MLP head on the same ResNet-Mamba backbone, Vim-S vs each hybrid, Vim-S vs its second initialisation (run-to-run variation).
  * Reference-model comparisons (section D3; also exploratory, added after the results, own Holm family of ten): the ResNet-50 and
    ViT-B/16 reference models of secondary_train.py (linear / MLP / KAN heads on ViT-B/16). They answer: does a KAN head help on a
    second backbone, is the ViT backbone better than the CNN, and how do Vim-S and the hybrids compare with the reference models.
    They are base models only: no ensemble contains them, so the plan's ensemble results (E1..E5) are unchanged.

Usage:  python ensemble_eval.py --run_dir . [--out_dir results/ensemble_v2]
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import os
import sys
from contextlib import redirect_stdout

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, recall_score

PROTOCOL_TAG = "v2"
SEEDS = [42, 1024, 2026]
FOLDS = [1, 2, 3, 4, 5]
BASE = {"vim": "vision_mamba", "kan": "resnet_mamba_kan", "mlp": "resnet_mamba_mlp", "vim2": "vision_mamba_ctrl",
        # reference models of secondary_train.py (same protocol, same splits); not part of the ensembles E1..E5
        "r50": "resnet50", "vit": "vit_b16", "vitmlp": "vit_b16_mlp", "vitkan": "vit_b16_kan"}
LABEL = {"vim": "Vim-S", "kan": "ResNet-Mamba-KAN", "mlp": "ResNet-Mamba-MLP", "vim2": "Vim-S (2nd init)",
         "r50": "ResNet-50 (linear head)", "vit": "ViT-B/16 (linear head)", "vitmlp": "ViT-B/16 + MLP head", "vitkan": "ViT-B/16 + KAN head"}
ENSEMBLES = {                       # fixed by the plan (section 3)
    "E1": ["vim", "kan"],           # primary
    "E2": ["vim", "mlp"],
    "E3": ["vim", "vim2"],
    "E4": ["kan", "mlp"],
    "E5": ["vim", "kan", "mlp"],
}


# ---------------------------------------------------------------------------------------------
# Loading and consistency checks
# ---------------------------------------------------------------------------------------------
def load_runs(run_dir, seeds, folds):
    """runs[system][(seed, fold)] = dict(probs, labels, val_idx); only base models."""
    runs = {k: {} for k in BASE}
    classes = None
    for key, name in BASE.items():
        for s, f in itertools.product(seeds, folds):
            p = os.path.join(run_dir, "preds", f"{name}_{PROTOCOL_TAG}_seed_{s}_fold_{f}.npz")
            if not os.path.isfile(p):
                continue
            z = np.load(p, allow_pickle=False)
            runs[key][(s, f)] = dict(probs=z["probs"].astype(np.float64), labels=z["labels"].astype(int),
                                     val_idx=z["val_idx"].astype(int))
            cn = [str(c) for c in z["class_names"]]
            if classes is None:
                classes = cn
            elif classes != cn:
                raise SystemExit(f"Class names differ between prediction files ({p}).")
    return runs, classes


def check_consistency(runs, run_dir, seeds, folds):
    """Same validation images + labels for every model in every (seed, fold); agree with splits_seed_*.json."""
    problems = []
    for s, f in itertools.product(seeds, folds):
        have = [k for k in runs if (s, f) in runs[k]]
        for a, b in itertools.combinations(have, 2):
            ra, rb = runs[a][(s, f)], runs[b][(s, f)]
            if not np.array_equal(ra["val_idx"], rb["val_idx"]):
                problems.append(f"seed {s} fold {f}: validation indices of {BASE[a]} and {BASE[b]} differ")
            elif not np.array_equal(ra["labels"], rb["labels"]):
                problems.append(f"seed {s} fold {f}: labels of {BASE[a]} and {BASE[b]} differ")
    for s in seeds:
        sp = os.path.join(run_dir, "results", f"splits_seed_{s}.json")
        if not os.path.isfile(sp):
            print(f"[!] {sp} not found - split consistency with the saved split file not checked for seed {s}.")
            continue
        with open(sp, encoding="utf-8") as fh:
            ref = json.load(fh)["val_idx"]
        for f in folds:
            for k in runs:
                if (s, f) in runs[k] and not np.array_equal(np.sort(runs[k][(s, f)]["val_idx"]), np.sort(ref[f - 1])):
                    problems.append(f"seed {s} fold {f}: {BASE[k]} validation indices differ from {sp}")
    if problems:
        raise SystemExit("INCONSISTENT RUNS - analysis stopped (plan section 2):\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------------------------
# Systems = base models + ensembles
# ---------------------------------------------------------------------------------------------
def build_systems(runs):
    """systems[name] = dict(members=[...], runs={(seed, fold): dict(probs, labels, val_idx)}). Ensemble = mean of probabilities."""
    systems = {}
    for k in BASE:
        if runs[k]:
            systems[k] = dict(members=[k], runs=runs[k], label=LABEL[k])
    for e, members in ENSEMBLES.items():
        if not all(runs[m] for m in members):
            print(f"[!] {e} ({' + '.join(LABEL[m] for m in members)}): at least one member has no runs - not computed.")
            continue
        common = sorted(set.intersection(*[set(runs[m]) for m in members]))
        if not common:
            print(f"[!] {e}: no (seed, fold) with all members - not computed.")
            continue
        out = {}
        for key in common:
            probs = np.mean([runs[m][key]["probs"] for m in members], axis=0)   # unweighted mean, nothing fitted
            out[key] = dict(probs=probs, labels=runs[members[0]][key]["labels"], val_idx=runs[members[0]][key]["val_idx"])
        systems[e] = dict(members=members, runs=out, label=f"{e}: " + " + ".join(LABEL[m] for m in members))
    return systems


def predictions(r):
    return np.argmax(r["probs"], axis=1)          # ties -> lowest class index


def run_metrics(r, n_classes):
    y, p = r["labels"], predictions(r)
    labs = list(range(n_classes))
    return dict(acc=float((y == p).mean() * 100),
                f1=float(f1_score(y, p, labels=labs, average="macro", zero_division=0) * 100),
                recall=(recall_score(y, p, labels=labs, average=None, zero_division=0) * 100).tolist(),
                n_val=int(len(y)))


def sd(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------
def nadeau_bengio(diffs, n_val_mean, n_train_mean):
    """Corrected resampled paired t-test: var multiplied by (1/J + n_test/n_train)."""
    d = np.asarray(diffs, dtype=float)
    J = len(d)
    if J < 2:
        return dict(J=J, mean=float(d.mean()) if J else float("nan"), t=float("nan"), p=float("nan"), lo=float("nan"), hi=float("nan"))
    var = d.var(ddof=1)
    se = float(np.sqrt((1.0 / J + n_val_mean / n_train_mean) * var))
    mean = float(d.mean())
    if se == 0.0:
        return dict(J=J, mean=mean, t=float("nan"), p=1.0 if mean == 0 else 0.0, lo=mean, hi=mean)
    t = mean / se
    p = float(2 * stats.t.sf(abs(t), J - 1))
    h = float(stats.t.ppf(0.975, J - 1) * se)
    return dict(J=J, mean=mean, t=float(t), p=p, lo=mean - h, hi=mean + h)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def pooled_seed(system, seed, folds):
    """Concatenate the seed's folds in image-index order -> (val_idx, labels, preds); None if a fold is missing."""
    keys = [(seed, f) for f in folds]
    if not all(k in system["runs"] for k in keys):
        return None
    idx = np.concatenate([system["runs"][k]["val_idx"] for k in keys])
    y = np.concatenate([system["runs"][k]["labels"] for k in keys])
    p = np.concatenate([predictions(system["runs"][k]) for k in keys])
    o = np.argsort(idx)
    return idx[o], y[o], p[o]


def mcnemar_exact(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b))      # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))      # A wrong, B right
    n = b + c
    p = 1.0 if n == 0 else float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    return b, c, p


def paired_runs(sa, sb, metric="acc", n_classes=6):
    keys = sorted(set(sa["runs"]) & set(sb["runs"]))
    a = np.array([run_metrics(sa["runs"][k], n_classes)[metric] for k in keys])
    b = np.array([run_metrics(sb["runs"][k], n_classes)[metric] for k in keys])
    return keys, a, b


# ---------------------------------------------------------------------------------------------
def analyse(args):
    runs, classes = load_runs(args.run_dir, args.seeds, FOLDS)
    if classes is None:
        raise SystemExit(f"No prediction files found under {os.path.join(args.run_dir, 'preds')}")
    n_classes = len(classes)
    check_consistency(runs, args.run_dir, args.seeds, FOLDS)
    systems = build_systems(runs)

    print("=" * 100)
    print("ENSEMBLE ANALYSIS - epoch-20 predictions, equal-weight soft voting, nothing fitted on validation data")
    print("=" * 100)
    print("Runs found: " + ", ".join(f"{LABEL[k]} {len(runs[k])}" for k in BASE))
    print("Consistency check passed: identical validation indices and labels for all models in every (seed, fold).")

    # --- per run metrics ------------------------------------------------------------------
    metrics = {name: {k: run_metrics(r, n_classes) for k, r in sy["runs"].items()} for name, sy in systems.items()}
    with open(os.path.join(args.out_dir, "ensemble_runs.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "seed", "fold", "n_val", "accuracy", "macro_f1"] + [f"recall_{c}" for c in classes])
        for name in systems:
            for (s, f), m in sorted(metrics[name].items()):
                w.writerow([name, s, f, m["n_val"], f"{m['acc']:.4f}", f"{m['f1']:.4f}"] + [f"{x:.4f}" for x in m["recall"]])

    # --- summary --------------------------------------------------------------------------
    print("\n" + "-" * 100)
    print("A. ACCURACY AND MACRO-F1 (mean +- sample SD (n-1), min, max over runs)")
    print("-" * 100)
    print(f"{'system':58s} {'runs':>4s} {'Accuracy %':>15s} {'min':>6s} {'max':>6s} {'macro-F1 %':>15s}")
    rows = []
    for name, sy in systems.items():
        a = np.array([m["acc"] for m in metrics[name].values()])
        f1 = np.array([m["f1"] for m in metrics[name].values()])
        print(f"{sy['label']:58s} {len(a):4d} {a.mean():8.2f} +- {sd(a):4.2f} {a.min():6.2f} {a.max():6.2f} {f1.mean():8.2f} +- {sd(f1):4.2f}")
        rows.append([name, sy["label"], len(a), a.mean(), sd(a), a.min(), a.max(), f1.mean(), sd(f1)])
    with open(os.path.join(args.out_dir, "ensemble_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "label", "runs", "acc_mean", "acc_sd", "acc_min", "acc_max", "f1_mean", "f1_sd"])
        w.writerows([[r[0], r[1], r[2]] + [f"{x:.4f}" for x in r[3:]] for r in rows])

    print("\n" + "-" * 100)
    print("B. PER-CLASS RECALL (mean over runs, %)")
    print("-" * 100)
    print(f"{'system':58s} " + " ".join(f"{c[:9]:>9s}" for c in classes))
    for name, sy in systems.items():
        rec = np.mean([m["recall"] for m in metrics[name].values()], axis=0)
        print(f"{sy['label']:58s} " + " ".join(f"{x:9.1f}" for x in rec))

    # --- pooled confusion matrices (per seed) ----------------------------------------------
    cms = {}
    for name, sy in systems.items():
        for s in args.seeds:
            pooled = pooled_seed(sy, s, FOLDS)
            if pooled is None:
                continue
            _, y, p = pooled
            cms.setdefault(name, {})[str(s)] = confusion_matrix(y, p, labels=list(range(n_classes))).tolist()
    with open(os.path.join(args.out_dir, "confusion_matrices.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(classes=classes, note="pooled over the 5 folds of one seed; rows = true, columns = predicted", matrices=cms), fh, indent=1)
    print(f"\n(Pooled confusion matrices per seed -> {os.path.join(args.out_dir, 'confusion_matrices.json')})")

    # --- error overlap ---------------------------------------------------------------------
    print("\n" + "-" * 100)
    print("C. ERROR OVERLAP OF THE BASE MODELS (over all seed x image predictions)")
    print("-" * 100)
    print(f"{'pair':44s} {'both wrong %':>13s} {'only 1st %':>11s} {'only 2nd %':>11s} {'oracle acc %':>13s}")
    ov_rows = []
    for a, b in itertools.combinations([k for k in BASE if k in systems], 2):
        keys = sorted(set(systems[a]["runs"]) & set(systems[b]["runs"]))
        if not keys:
            continue
        wa = np.concatenate([predictions(systems[a]["runs"][k]) != systems[a]["runs"][k]["labels"] for k in keys])
        wb = np.concatenate([predictions(systems[b]["runs"][k]) != systems[b]["runs"][k]["labels"] for k in keys])
        both, only_a, only_b = (wa & wb).mean() * 100, (wa & ~wb).mean() * 100, (~wa & wb).mean() * 100
        oracle = (~(wa & wb)).mean() * 100
        print(f"{LABEL[a] + ' / ' + LABEL[b]:44s} {both:13.2f} {only_a:11.2f} {only_b:11.2f} {oracle:13.2f}")
        ov_rows.append([a, b, len(keys), both, only_a, only_b, oracle])
    with open(os.path.join(args.out_dir, "error_overlap.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model_1", "model_2", "runs", "both_wrong_pct", "only_1_wrong_pct", "only_2_wrong_pct", "oracle_acc_pct"])
        w.writerows([[r[0], r[1], r[2]] + [f"{x:.4f}" for x in r[3:]] for r in ov_rows])
    print("(oracle = share of images that at least one of the two models classifies correctly: upper bound of any 2-model vote)")

    # --- tests -----------------------------------------------------------------------------
    print("\n" + "-" * 100)
    print("D. STATISTICAL TESTS (accuracy)")
    print("-" * 100)
    ref = metrics["vim"] if "vim" in metrics else next(iter(metrics.values()))
    n_total = {s: sum(m["n_val"] for (ss, _), m in ref.items() if ss == s) for s in args.seeds}
    n_val_mean = float(np.mean([m["n_val"] for m in ref.values()]))
    n_train_mean = float(np.mean([n_total[s] - m["n_val"] for (s, _), m in ref.items()]))
    print(f"Nadeau-Bengio correction factor 1/J + n_val/n_train with n_val ~ {n_val_mean:.1f}, n_train ~ {n_train_mean:.1f}")

    test_rows = []

    def compare(tag, a_name, b_name, family):
        """a minus b."""
        if a_name not in systems or b_name not in systems:
            print(f"[!] {tag}: {a_name} or {b_name} not available.")
            return None
        keys, a, b = paired_runs(systems[a_name], systems[b_name], "acc", n_classes)
        res = nadeau_bengio(a - b, n_val_mean, n_train_mean)
        res.update(tag=tag, a=a_name, b=b_name, family=family, runs=len(keys))
        # McNemar per seed
        res["mcnemar"] = {}
        for s in args.seeds:
            pa, pb = pooled_seed(systems[a_name], s, FOLDS), pooled_seed(systems[b_name], s, FOLDS)
            if pa is None or pb is None:
                continue
            assert np.array_equal(pa[0], pb[0]) and np.array_equal(pa[1], pb[1])
            res["mcnemar"][s] = mcnemar_exact(pa[2] == pa[1], pb[2] == pb[1])
        test_rows.append(res)
        return res

    primary = compare("PRIMARY  E1 vs Vim-S", "E1", "vim", "primary")
    secondary_specs = []
    for e in ("E2", "E3", "E4", "E5"):
        if e in systems:
            members = ENSEMBLES[e]
            best = max(members, key=lambda m: np.mean([x["acc"] for x in metrics[m].values()]))
            secondary_specs.append((f"{e} vs its best member ({LABEL[best]})", e, best))
    secondary_specs.append(("E1 vs E2 (KAN vs MLP partner)", "E1", "E2"))
    secondary_specs.append(("E1 vs E3 (different architecture vs 2nd Vim)", "E1", "E3"))
    secondary = [r for r in (compare(t, a, b, "secondary") for t, a, b in secondary_specs) if r is not None]
    if secondary:
        adj = holm([r["p"] for r in secondary])
        for r, q in zip(secondary, adj):
            r["p_holm"] = q

    def show(r):
        print(f"\n{r['tag']}   [{LABEL.get(r['a'], r['a']) if r['a'] in LABEL else systems[r['a']]['label']}  minus  "
              f"{LABEL.get(r['b'], r['b']) if r['b'] in LABEL else systems[r['b']]['label']}]")
        print(f"  paired runs J = {r['runs']} | mean accuracy difference {r['mean']:+.2f} points | "
              f"95% interval [{r['lo']:+.2f}, {r['hi']:+.2f}] | corrected t = {r['t']:.2f} | p = {r['p']:.4f}"
              + (f" | Holm-adjusted p = {r['p_holm']:.4f}" if "p_holm" in r else ""))
        if r["mcnemar"]:
            print("  McNemar exact, per seed (a-only-correct / b-only-correct, p): " +
                  "; ".join(f"seed {s}: {b}/{c}, p={p:.4f}" for s, (b, c, p) in r["mcnemar"].items()))

    if primary:
        show(primary)
    for r in secondary:
        show(r)

    # --- D2: comparisons of the single base models (added after the results; exploratory; own Holm family) -----------------
    base_specs = [("KAN vs MLP head (same ResNet-Mamba backbone)", "kan", "mlp"),
                  ("Vim-S vs ResNet-Mamba-KAN", "vim", "kan"),
                  ("Vim-S vs ResNet-Mamba-MLP", "vim", "mlp"),
                  ("Vim-S vs Vim-S (2nd init): run-to-run variation of the identical recipe", "vim", "vim2")]
    base = [r for r in (compare(t, a, b, "base_models") for t, a, b in base_specs) if r is not None]
    if base:
        for r, q in zip(base, holm([r["p"] for r in base])):
            r["p_holm"] = q
        print("\n" + "-" * 100)
        print("D2. BASE-MODEL COMPARISONS (single models; added after the results at the request of reviewer 2 - exploratory; Holm over these four)")
        print("-" * 100)
        for r in base:
            show(r)

    # --- D3: reference models (ResNet-50, ViT-B/16 with linear / MLP / KAN head); exploratory; own Holm family ------------------
    ref_specs = [("ViT-B/16: KAN vs MLP head", "vitkan", "vitmlp"),
                 ("ViT-B/16: KAN vs linear head", "vitkan", "vit"),
                 ("ViT-B/16: MLP vs linear head", "vitmlp", "vit"),
                 ("ViT-B/16 (linear) vs ResNet-50 (linear): transformer vs CNN backbone", "vit", "r50"),
                 ("Vim-S vs ResNet-50 (linear)", "vim", "r50"),
                 ("Vim-S vs ViT-B/16 (linear)", "vim", "vit"),
                 ("Vim-S vs ViT-B/16 + MLP", "vim", "vitmlp"),
                 ("Vim-S vs ViT-B/16 + KAN", "vim", "vitkan"),
                 ("MLP head, ResNet-Mamba vs ViT-B/16 backbone", "mlp", "vitmlp"),
                 ("KAN head, ResNet-Mamba vs ViT-B/16 backbone", "kan", "vitkan")]
    ref_tests = [r for r in (compare(t, a, b, "reference_models") for t, a, b in ref_specs) if r is not None]
    if ref_tests:
        for r, q in zip(ref_tests, holm([r["p"] for r in ref_tests])):
            r["p_holm"] = q
        print("\n" + "-" * 100)
        print(f"D3. REFERENCE-MODEL COMPARISONS (ResNet-50, ViT-B/16; added after the results - exploratory; Holm over these {len(ref_tests)})")
        print("-" * 100)
        for r in ref_tests:
            show(r)

    with open(os.path.join(args.out_dir, "ensemble_tests.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "comparison", "a", "b", "runs", "mean_diff_pts", "ci95_lo", "ci95_hi", "t_corrected", "p", "p_holm"] +
                   [f"mcnemar_seed_{s}_a_only,b_only,p" for s in args.seeds])
        for r in test_rows:
            w.writerow([r["family"], r["tag"], r["a"], r["b"], r["runs"], f"{r['mean']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}",
                        f"{r['t']:.4f}", f"{r['p']:.6f}", f"{r.get('p_holm', float('nan')):.6f}"] +
                       [("%d,%d,%.6f" % r["mcnemar"][s]) if s in r["mcnemar"] else "" for s in args.seeds])

    # --- verdict according to the plan ----------------------------------------------------
    print("\n" + "-" * 100)
    print("E. VERDICT ACCORDING TO THE PLAN (section 7)")
    print("-" * 100)
    if primary is None:
        print("Primary comparison could not be computed (E1 or Vim-S missing).")
    elif primary["p"] < 0.05 and primary["mean"] > 0:
        print(f"E1 improves on Vision Mamba alone: {primary['mean']:+.2f} points, 95% interval [{primary['lo']:+.2f}, {primary['hi']:+.2f}], p = {primary['p']:.4f}.")
    elif primary["p"] < 0.05 and primary["mean"] < 0:
        print(f"E1 is WORSE than Vision Mamba alone: {primary['mean']:+.2f} points, p = {primary['p']:.4f}.")
    else:
        print(f"No significant gain of E1 over Vision Mamba alone ({primary['mean']:+.2f} points, 95% interval "
              f"[{primary['lo']:+.2f}, {primary['hi']:+.2f}], p = {primary['p']:.4f}). Report spread, recall and error overlap as descriptive only.")
    if primary:
        sd_e1 = sd([m["acc"] for m in metrics["E1"].values()])
        sd_v = sd([m["acc"] for m in metrics["vim"].values()])
        print(f"Spread of run accuracy: E1 SD {sd_e1:.2f} vs Vim-S SD {sd_v:.2f}.")
    kan_mlp = next((r for r in base if r["a"] == "kan" and r["b"] == "mlp"), None) if base else None
    if kan_mlp:
        verdict = "a significant" if kan_mlp["p"] < 0.05 else "no significant"
        print(f"KAN vs MLP head on the same backbone: {kan_mlp['mean']:+.2f} points, 95% interval [{kan_mlp['lo']:+.2f}, {kan_mlp['hi']:+.2f}], "
              f"p = {kan_mlp['p']:.4f} (Holm over the four base-model comparisons: {kan_mlp['p_holm']:.4f}) -> {verdict} difference (exploratory test).")
    vit_kan_mlp = next((r for r in ref_tests if r["a"] == "vitkan" and r["b"] == "vitmlp"), None) if ref_tests else None
    if vit_kan_mlp:
        verdict = "a significant" if vit_kan_mlp["p"] < 0.05 else "no significant"
        print(f"KAN vs MLP head on ViT-B/16 (second backbone): {vit_kan_mlp['mean']:+.2f} points, 95% interval [{vit_kan_mlp['lo']:+.2f}, "
              f"{vit_kan_mlp['hi']:+.2f}], p = {vit_kan_mlp['p']:.4f} (Holm over the {len(ref_tests)} reference-model comparisons: "
              f"{vit_kan_mlp['p_holm']:.4f}) -> {verdict} difference (exploratory test).")
    if "E3" not in systems:
        print("E3 (two Vim-S) is NOT done: report it as not done (plan section 3).")
    print("Limitation to state with these numbers: image-level splits; tile/slab independence is not established. "
          "15 runs share images across seeds/folds, so the corrected t-test is approximate.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ensemble analysis according to the pre-specified plan")
    ap.add_argument("--run_dir", default=".", help="directory containing preds/ and results/")
    ap.add_argument("--out_dir", default=None, help="default: <run_dir>/results/ensemble_v2")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args(argv)
    args.out_dir = args.out_dir or os.path.join(args.run_dir, "results", "ensemble_v2")
    os.makedirs(args.out_dir, exist_ok=True)
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

        def flush(self):
            sys.__stdout__.flush()

    try:
        with redirect_stdout(Tee()):
            analyse(args)
    finally:
        with open(os.path.join(args.out_dir, "ensemble_report.txt"), "w", encoding="utf-8") as fh:
            fh.write(buf.getvalue())
    print(f"\nFiles written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
