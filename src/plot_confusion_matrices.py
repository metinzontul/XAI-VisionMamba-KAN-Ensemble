"""
Author: Prof. Dr. Metin Zontul
Project: XAI-VisionMamba-KAN-Ensemble
File: plot_confusion_matrices.py

Confusion matrices from the STORED epoch-20 predictions (preds/*.npz). No model is loaded, no GPU is needed, and
no seed/fold is selected: every one of the 15 runs (3 seeds x 5 folds) enters, so each validation image is
counted once per seed (3 x 996 = 2988 predictions). The ensemble is formed exactly as in ensemble_eval.py
(unweighted mean of the members' softmax probabilities, arg-max).

Default figure = Figure 3 of the manuscript:  (a) Vim-S   (b) ResNet-Mamba-KAN   (c) E1 = Vim-S + ResNet-Mamba-KAN.

Usage:
    python plot_confusion_matrices.py --run_dir .
    python plot_confusion_matrices.py --run_dir . --systems vim kan E1 E3 --per_seed

--systems accepts the keys of ensemble_eval.py: vim kan mlp vim2 (single models) and E1..E5 (ensembles).

Paper panels (default on, --paper_dir): one file per matrix, 600 dpi, no panel letters on the figure (the letter is in the
file name): <paper_dir>/article/Fig3a_confusion_vim_s.png/.pdf, Fig3b_confusion_hybrid_mlp, Fig3c_confusion_hybrid_kan,
Fig3d_confusion_ensemble_E1  (--paper_systems vim mlp kan E1).

Outputs (out_dir, default <run_dir>/results/figures):
    Figure_confusion_matrices.png / .pdf        counts (and row-normalised %) aggregated over all runs
    Figure_confusion_matrix_seed_<s>.png        one figure per seed, with --per_seed (each image once)
    confusion_matrices_aggregated.csv           the counts of every system
    confusion_matrices_report.txt               accuracy, macro-F1 and per-class recall of the aggregated matrices
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

import ensemble_eval as ev
import figio

SUB = "abcdefgh"
SLUG = {"vim": "vim_s", "mlp": "hybrid_mlp", "kan": "hybrid_kan", "vim2": "vim_s_second_init",
        "E1": "ensemble_E1", "E2": "ensemble_E2", "E3": "ensemble_E3", "E4": "ensemble_E4", "E5": "ensemble_E5"}


def metrics_from_cm(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    rec = tp / np.maximum(cm.sum(1), 1)
    prec = tp / np.maximum(cm.sum(0), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return dict(acc=float(tp.sum() / cm.sum() * 100), macro_f1=float(f1.mean() * 100), recall=(rec * 100).tolist(), n=int(cm.sum()))


def confusion(system, n_classes, seeds=None):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    used = 0
    for (s, f), r in sorted(system["runs"].items()):
        if seeds is not None and s not in seeds:
            continue
        y, p = r["labels"], ev.predictions(r)
        np.add.at(cm, (y, p), 1)
        used += 1
    return cm, used


def draw(ax, cm, classes, title, mathtext_tag=None):
    import matplotlib.pyplot as plt
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    n = len(classes)
    for i in range(n):
        for j in range(n):
            col = "white" if norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j] * 100:.1f}%", ha="center", va="center", color=col,
                    fontsize=8.5, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted class", fontweight="bold"); ax.set_ylabel("True class", fontweight="bold")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True); ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0)
    return im


def make_figure(cms, labels, classes, out_png, out_pdf, dpi, footer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif"})
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 6.6), squeeze=False)
    for i, (ax, cm, lab) in enumerate(zip(axes[0], cms, labels)):
        m = metrics_from_cm(cm)
        draw(ax, cm, classes, f"({SUB[i]}) {lab}\naccuracy {m['acc']:.2f}%, macro-F1 {m['macro_f1']:.2f}%, n = {m['n']}")
    fig.text(0.5, 0.008, footer, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Confusion matrices from the stored epoch-20 predictions")
    ap.add_argument("--run_dir", default=".")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--systems", nargs="+", default=["vim", "kan", "E1"])
    ap.add_argument("--seeds", type=int, nargs="+", default=ev.SEEDS)
    ap.add_argument("--per_seed", action="store_true", help="additionally one figure per seed (each image counted once)")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--paper_dir", default=None, help="folder of the single-panel manuscript figures (default <run_dir>/results/paper_figures); 'none' to skip")
    ap.add_argument("--paper_systems", nargs="+", default=["vim", "mlp", "kan", "E1"], help="systems of the manuscript panels, in panel order (a, b, c, ...)")
    ap.add_argument("--paper_dpi", type=int, default=figio.PAPER_DPI)
    a = ap.parse_args(argv)
    out = a.out_dir or os.path.join(a.run_dir, "results", "figures")
    os.makedirs(out, exist_ok=True)

    runs, classes = ev.load_runs(a.run_dir, a.seeds, ev.FOLDS)
    if classes is None:
        raise SystemExit(f"no prediction files under {os.path.join(a.run_dir, 'preds')}")
    ev.check_consistency(runs, a.run_dir, a.seeds, ev.FOLDS)
    systems = ev.build_systems(runs)
    missing = [s for s in a.systems if s not in systems]
    if missing:
        raise SystemExit(f"system(s) not available (missing prediction files?): {missing}")

    labels = [systems[s]["label"].split(": ")[-1] if s.startswith("E") is False else systems[s]["label"] for s in a.systems]
    n_classes = len(classes)
    cms, runs_used, report = [], [], []
    for s in a.systems:
        cm, used = confusion(systems[s], n_classes, a.seeds)
        cms.append(cm); runs_used.append(used)
        m = metrics_from_cm(cm)
        report += [f"{systems[s]['label']}  ({used} runs, {m['n']} predictions)",
                   f"  accuracy {m['acc']:.2f}% | macro-F1 {m['macro_f1']:.2f}% (computed from the aggregated matrix)",
                   "  recall: " + ", ".join(f"{c} {r:.1f}" for c, r in zip(classes, m["recall"])), ""]
    if len(set(runs_used)) > 1:
        report.append(f"[!] the systems rest on different numbers of runs: {dict(zip(a.systems, runs_used))}")

    footer = (f"Aggregated over all {max(runs_used)} runs (seeds {', '.join(map(str, a.seeds))} x 5 folds, epoch-20 weights); every validation image is "
              f"counted once per seed. Cells: count and row-normalised percentage.")
    make_figure(cms, labels, classes, os.path.join(out, "Figure_confusion_matrices.png"),
                os.path.join(out, "Figure_confusion_matrices.pdf"), a.dpi, footer)

    if a.per_seed:
        for sd_ in a.seeds:
            per = [confusion(systems[s], n_classes, [sd_])[0] for s in a.systems]
            make_figure(per, labels, classes, os.path.join(out, f"Figure_confusion_matrix_seed_{sd_}.png"), None, a.dpi,
                        f"Seed {sd_}: 5 folds pooled (each of the 996 images once), epoch-20 weights.")

    with open(os.path.join(out, "confusion_matrices_aggregated.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "true_class"] + [f"pred_{c}" for c in classes])
        for s, cm in zip(a.systems, cms):
            for i, c in enumerate(classes):
                w.writerow([s, c] + cm[i].tolist())
    paper_dir = a.paper_dir or os.path.join(a.run_dir, "results", "paper_figures")
    if paper_dir.lower() != "none":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif"})
        art, _ = figio.paper_dirs(paper_dir)
        print("\nPaper panels:")
        for k, sname in enumerate(a.paper_systems):
            if sname not in systems:
                print(f"[!] system {sname} not available - panel omitted"); continue
            cm, used = confusion(systems[sname], n_classes, a.seeds)
            m = metrics_from_cm(cm)
            lab = systems[sname]["label"] if sname.startswith("E") else systems[sname]["label"].split(": ")[-1]
            fig, ax = plt.subplots(figsize=(6.4, 6.6))
            draw(ax, cm, classes, f"{lab}\naccuracy {m['acc']:.2f}%, macro-F1 {m['macro_f1']:.2f}%, n = {m['n']}")
            fig.tight_layout()
            figio.save_panel(fig, art, f"Fig3{figio.LETTERS[k]}_confusion_{SLUG.get(sname, sname)}", a.paper_dpi, pdf=True)
            plt.close(fig)
    text = "\n".join(report)
    with open(os.path.join(out, "confusion_matrices_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"Files written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
