"""
Author: Prof. Dr. Metin Zontul
Project: XAI-VisionMamba-KAN-Ensemble
File: plot_learning_curves.py

Learning curves (mean +- sample SD over the 15 seed-fold runs) drawn from the RAW per-epoch lines of the
master training logs written by master_train.py. Nothing is typed in by hand and nothing is smoothed:

    Epoch [e/20] - Train Loss: ..., Train Acc: ...% | Val Loss: ..., Val Acc: ...% [..s]

Replaces the earlier script, whose curves were hard-coded arrays ("simulated mean/std profiles").

Usage:
    python plot_learning_curves.py --logs logs/MASTER_TRAINING_LOG_*.txt
    python plot_learning_curves.py --logs logs/*.txt --models vision_mamba resnet_mamba_kan --out_dir results/figures

Several log files can be given (the hybrids and Vim were trained in separate runs). If the same
(model, seed, fold) occurs in several logs, the LAST one given wins and a warning is printed.
Only runs that completed all epochs are used; the number of runs per model is printed and written into the figure.

Paper panels (default on, --paper_dir): every panel as its own file, 600 dpi, no panel letters on the figure:
    <paper_dir>/article/Fig2a_learning_curve_vim_s.png/.pdf, Fig2b_..._hybrid_mlp, Fig2c_..._hybrid_kan
    <paper_dir>/supplementary/FigS1a_val_acc_runs_vim_s.png, FigS1b_..., FigS1c_...   (every single run)

Outputs (out_dir):
    Figure_learning_curves.png / .pdf     mean +- SD panels (loss on the left axis, validation accuracy on the right)
    Figure_val_acc_individual_runs.png    every single run's validation accuracy (shows epoch-to-epoch volatility)
    learning_curves_summary.csv           per-epoch mean and SD of every quantity
    learning_curves_report.txt            final-epoch statistics and the best-epoch bias
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

import figio

EPOCH_RE = re.compile(r"Epoch \[(\d+)/(\d+)\] - Train Loss: ([0-9.eE+-]+), Train Acc: ([0-9.]+)% \| Val Loss: ([0-9.eE+-]+), Val Acc: ([0-9.]+)%")
START_RE = re.compile(r"STARTING: (\w+) \| seed (\d+) \| fold (\d+)/(\d+)")
DONE_RE = re.compile(r"-> Fold (\d+) Completed\.")

TITLE = {"vision_mamba": "Vision Mamba (Vim-S)", "resnet_mamba_kan": "ResNet-50 + Mamba + KAN",
         "resnet_mamba_mlp": "ResNet-50 + Mamba + MLP", "vision_mamba_ctrl": "Vision Mamba (Vim-S), 2nd initialisation"}


SLUG = {"vision_mamba": "vim_s", "resnet_mamba_kan": "hybrid_kan", "resnet_mamba_mlp": "hybrid_mlp", "vision_mamba_ctrl": "vim_s_second_init"}
C_TR, C_VL, C_ACC = "#1f77b4", "#ff7f0e", "#d62728"
SEED_COL = {42: "#1b9e77", 1024: "#d95f02", 2026: "#7570b3"}


def draw_curve(ax, mname, arr):
    """One panel: mean +- SD of train loss, val loss (left axis) and val accuracy (right axis)."""
    E = arr.shape[1]
    ep = np.arange(1, E + 1)
    trl, tra, vl, va = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    axr = ax.twinx()
    l1, = ax.plot(ep, trl.mean(0), color=C_TR, ls="-", marker="o", ms=3.5, label="Train loss (mean)")
    l2, = ax.plot(ep, vl.mean(0), color=C_VL, ls="--", marker="s", ms=3.5, label="Val loss (mean)")
    l3, = axr.plot(ep, va.mean(0), color=C_ACC, ls="-.", marker="D", ms=3.5, label="Val accuracy (mean)")
    ax.fill_between(ep, np.clip(trl.mean(0) - sd(trl), 0, None), trl.mean(0) + sd(trl), color=C_TR, alpha=0.2)
    ax.fill_between(ep, np.clip(vl.mean(0) - sd(vl), 0, None), vl.mean(0) + sd(vl), color=C_VL, alpha=0.2)
    axr.fill_between(ep, va.mean(0) - sd(va), va.mean(0) + sd(va), color=C_ACC, alpha=0.15)
    ax.axvline(E, color="grey", ls=":", lw=1)
    ax.set_title(f"{TITLE.get(mname, mname)}\n{arr.shape[0]} runs (mean ± SD)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (focal)")
    axr.set_ylabel("Validation accuracy (%)")
    ax.set_xticks(range(1, E + 1, 2))
    ax.set_ylim(0, 0.8)                                      # the same loss range in every panel
    axr.set_ylim(50, 100)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend([l1, l2, l3], [l.get_label() for l in (l1, l2, l3)], loc="center right")


def draw_individual(ax, mname, d):
    """One panel: validation accuracy of every single run, coloured by seed, and the mean of all runs."""
    ep = np.arange(1, d["arr"].shape[1] + 1)
    for k, arr in zip(d["keys"], d["arr"]):
        ax.plot(ep, arr[:, 3], color=SEED_COL.get(k[1], "k"), alpha=0.45, lw=1)
    ax.plot(ep, d["arr"][..., 3].mean(0), color="black", lw=2.2, label="mean of all runs")
    for sd_, c in SEED_COL.items():
        ax.plot([], [], color=c, label=f"seed {sd_}")
    ax.axvline(ep[-1], color="grey", ls=":", lw=1)
    ax.set_title(f"{TITLE.get(mname, mname)}: validation accuracy of every run")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation accuracy (%)"); ax.set_ylim(50, 100)
    ax.set_xticks(range(1, len(ep) + 1, 2)); ax.grid(True, ls=":", alpha=0.6); ax.legend(loc="lower right")


def collect(runs, models, quiet=False):
    data = {}
    for mname in models:
        keys = sorted(k for k in runs if k[0] == mname)
        if not keys:
            print(f"[!] no completed runs of {mname} in the given logs - panel omitted")
            continue
        arr = np.stack([runs[k]["arr"] for k in keys])           # (runs, epochs, 4)
        data[mname] = dict(keys=keys, arr=arr)
        if not quiet:
            print(f"{mname}: {len(keys)} completed runs, {arr.shape[1]} epochs (seeds {sorted({k[1] for k in keys})})")
        if len(keys) != 15:
            print(f"[!] {mname}: {len(keys)} runs instead of 15 - the figure legend states the real number.")
    return data


def export_paper_panels(data, paper_dir, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13, "legend.fontsize": 9.5, "font.family": "serif"})
    art, sup = figio.paper_dirs(paper_dir)
    for i, (m, d) in enumerate(data.items()):
        fig, ax = plt.subplots(figsize=(7.6, 5.6))
        draw_curve(ax, m, d["arr"]); fig.tight_layout()
        figio.save_panel(fig, art, f"Fig2{figio.LETTERS[i]}_learning_curve_{SLUG.get(m, m)}", dpi, pdf=True); plt.close(fig)
    for i, (m, d) in enumerate(data.items()):
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        draw_individual(ax, m, d); fig.tight_layout()
        figio.save_panel(fig, sup, f"FigS1{figio.LETTERS[i]}_val_acc_every_run_{SLUG.get(m, m)}", dpi, pdf=True); plt.close(fig)


def parse_logs(paths):
    """runs[(model, seed, fold)] = dict(arr=(E,4) [train_loss, train_acc, val_loss, val_acc], log=path)"""
    runs = {}
    for path in paths:
        cur, buf = None, []
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = START_RE.search(line)
                if m:
                    cur, buf, total = (m.group(1), int(m.group(2)), int(m.group(3))), [], None
                    continue
                m = EPOCH_RE.search(line)
                if m and cur is not None:
                    e, E = int(m.group(1)), int(m.group(2))
                    if e == 1:
                        buf = []
                    buf.append((float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6))))
                    total = E
                    continue
                if DONE_RE.search(line) and cur is not None and buf and total is not None and len(buf) == total:
                    if cur in runs:
                        print(f"[!] {cur} found again in {os.path.basename(path)} - this one replaces the earlier copy "
                              f"({os.path.basename(runs[cur]['log'])}).")
                    runs[cur] = dict(arr=np.array(buf), log=path)
                    cur, buf = None, []
    return runs


def sd(a, axis=0):
    return np.std(a, axis=axis, ddof=1) if a.shape[axis] > 1 else np.full(a.shape[1 - axis], np.nan)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Learning curves from the raw master training logs")
    ap.add_argument("--logs", nargs="+", required=True, help="master log files (globs allowed)")
    ap.add_argument("--models", nargs="+", default=["vision_mamba", "resnet_mamba_kan"],
                    help="models to draw, in panel order (default: the two panels of the manuscript)")
    ap.add_argument("--out_dir", default="results/figures")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--paper_dir", default="results/paper_figures", help="folder of the single-panel manuscript figures (article/, supplementary/); 'none' to skip")
    ap.add_argument("--paper_models", nargs="+", default=["vision_mamba", "resnet_mamba_mlp", "resnet_mamba_kan"],
                    help="models of the manuscript panels, in panel order (a, b, c)")
    ap.add_argument("--paper_dpi", type=int, default=figio.PAPER_DPI)
    ap.add_argument("--no_individual", action="store_true", help="skip the individual-runs figure")
    a = ap.parse_args(argv)

    paths = sorted({p for pat in a.logs for p in (glob.glob(pat) or [pat])})
    for p in paths:
        if not os.path.isfile(p):
            raise SystemExit(f"log file not found: {p}")
    runs = parse_logs(paths)
    os.makedirs(a.out_dir, exist_ok=True)

    data = collect(runs, a.models)
    if not data:
        raise SystemExit("nothing to draw")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13, "legend.fontsize": 9.5, "font.family": "serif"})
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(7.6 * n, 5.6), squeeze=False)
    report, rows = [], []
    for ax, (mname, d) in zip(axes[0], data.items()):
        arr = d["arr"]
        E = arr.shape[1]
        ep = np.arange(1, E + 1)
        trl, tra, vl, va = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
        draw_curve(ax, mname, arr)

        best_epoch_acc = va.max(1)                                 # per run: what "best checkpoint" reporting would give
        final = va[:, -1]
        report += [f"{TITLE.get(mname, mname)} ({arr.shape[0]} runs, {E} epochs)",
                   f"  final-epoch validation accuracy : {final.mean():.2f} ± {sd(final[:, None], 0)[0]:.2f}  (min {final.min():.2f}, max {final.max():.2f})",
                   f"  best-epoch validation accuracy  : {best_epoch_acc.mean():.2f} ± {sd(best_epoch_acc[:, None], 0)[0]:.2f}  "
                   f"(selection on the validation fold: optimistic by {best_epoch_acc.mean() - final.mean():+.2f} points)",
                   f"  mean per-epoch SD of val accuracy over epochs 11-{E}: {np.nanmean(sd(va)[10:]):.2f} points",
                   f"  mean val-accuracy change between consecutive epochs 11-{E} (absolute, per run): "
                   f"{np.abs(np.diff(va[:, 10:], axis=1)).mean():.2f} points", ""]
        for i, e in enumerate(ep):
            rows.append([mname, int(e), arr.shape[0]] + [f"{x:.5f}" for q in (trl, tra, vl, va) for x in (q[:, i].mean(), sd(q)[i])])

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out_dir, f"Figure_learning_curves.{ext}"), dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)

    if not a.no_individual:
        fig, axes = plt.subplots(1, n, figsize=(7.6 * n, 5.0), squeeze=False)
        for ax, (mname, d) in zip(axes[0], data.items()):
            draw_individual(ax, mname, d)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, "Figure_val_acc_individual_runs.png"), dpi=a.dpi, bbox_inches="tight")
        plt.close(fig)

    with open(os.path.join(a.out_dir, "learning_curves_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "epoch", "runs", "train_loss_mean", "train_loss_sd", "train_acc_mean", "train_acc_sd",
                    "val_loss_mean", "val_loss_sd", "val_acc_mean", "val_acc_sd"])
        w.writerows(rows)
    if a.paper_dir and a.paper_dir.lower() != "none":
        print("\nPaper panels:")
        export_paper_panels(collect(runs, a.paper_models, quiet=True), a.paper_dir, a.paper_dpi)
    text = "\n".join(report)
    with open(os.path.join(a.out_dir, "learning_curves_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\n" + text)
    print(f"Files written to {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
