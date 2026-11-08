"""
Author: Prof. Dr. Metin Zontul
Project: XAI-VisionMamba-KAN-Ensemble
File: xai_integrated_gradients.py

Integrated Gradients (IG) for the epoch-20 models, following section 8 of the Ensemble Analysis Plan.

What changed compared with the earlier script (get_xai_maps.py)
  * every map comes from the epoch-20 weights of the (seed, fold) in which the image is a VALIDATION image
    (the earlier script took files from the class folders, which may have been training images, and used the
    weights of different seeds/folds for different models);
  * images are not picked by hand or by os.listdir order: a seeded random draw of validation images per class,
    plus every image that at least one member of E1 misclassifies (all from seed 42);
  * the same method (IG, same baseline, same number of steps) for every architecture;
  * the maps are quantified (overlap, rank correlation, insertion/deletion curves, model-randomisation check);
  * the stored epoch-20 prediction of every image is re-computed and compared, as an integrity check.

IG settings (as in the manuscript, section 4.4): target = the model's predicted class logit, 50 Gauss-Legendre steps,
zero baseline in the normalised input space (= constant image with the ImageNet mean colour), positive attributions only
(channels summed, negative values set to 0).

Usage (needs captum:  pip install captum;  Vim needs the same environment as training, e.g. WSL):
    python xai_integrated_gradients.py --run_dir . --data "<path>/Marble"
    python xai_integrated_gradients.py --run_dir . --data ... --models vim kan mlp --per_image

Paper panels (default on, --paper_dir <run_dir>/results/paper_figures): every panel is its own file, 600 dpi, without panel letters on
the figure (the letter is part of the file name). Image figures are written column by column (same size, so the columns can be placed
side by side): Original | Vim-S | Hybrid-MLP | Hybrid-KAN.
    article/Fig4a..d_ig_main_*            4 images (first draw of Medium1, Medium2, Walnut1, Walnut2)
    article/Fig5a..d_ig_*                 IoU with chance level | insertion gain | deletion gain | model-randomisation check (PNG + PDF)
    supplementary/FigS2a..d_ig_classes_*  one image per class (6)
    supplementary/FigS3a..d_ig_misclassified_*

Outputs (out_dir, default <run_dir>/results/xai):
    Figure_IG_classes.png/.pdf         one randomly drawn validation image per class: original | IG of each model
    Figure_IG_misclassified.png/.pdf   images misclassified by at least one member of E1 (random subset)
    Figure_IG_metrics.png/.pdf         distributions of the quantitative measures
    xai_selection.csv                  every analysed image: path, class, fold, predictions of each model and of E1, reason
    xai_metrics.csv                    per image and model/pair measures
    xai_summary.txt                    summary statistics and integrity checks
    xai_maps_pooled.npz                28x28 pooled attribution maps
    per_image/                         (with --per_image) one panel per analysed image
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import itertools
import sys
import time

import numpy as np
import torch
from PIL import Image
from scipy import stats
from torchvision import datasets, transforms

import ensemble_eval as ev
import figio

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
IMG = 224
CELL = 8                                   # pooling cell: 224/8 = 28 x 28 grid used for overlap, correlation, curves
GRID = IMG // CELL
FILES = {"vim": "vision_mamba", "kan": "resnet_mamba_kan", "mlp": "resnet_mamba_mlp"}
NICE = {"vim": "Vision Mamba (Vim-S)", "kan": "ResNet-Mamba-KAN", "mlp": "ResNet-Mamba-MLP"}
SHORT = {"vim": "Vim-S", "kan": "Hybrid-KAN", "mlp": "Hybrid-MLP"}
ORDER = ("vim", "mlp", "kan")
SLUG = {"vim": "vim_s", "mlp": "hybrid_mlp", "kan": "hybrid_kan"}


# ----------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------
def build_model(kind, num_classes, weights=None, seed=None):
    """Architecture exactly as in training; `weights` = epoch-20 state dict; `seed` -> random (untrained) initialisation."""
    if seed is not None:
        torch.manual_seed(seed)
    if kind == "vim":
        from vision_mamba_train import create_vision_mamba
        model = create_vision_mamba(num_classes, variant="small", pretrained=False)
    elif kind == "kan":
        from resnet_mamba_kan import ResNetMambaKAN
        model = ResNetMambaKAN(num_classes, pretrained=False)
    elif kind == "mlp":
        from resnet_mamba_mlp import ResNetMambaMLP
        model = ResNetMambaMLP(num_classes, pretrained=False)
    else:
        raise ValueError(kind)
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model


# ----------------------------------------------------------------------------------------------
# images
# ----------------------------------------------------------------------------------------------
def load_image(path):
    """Same preprocessing as the stored validation predictions: RGB -> Resize((224,224)) -> /255 -> ImageNet normalisation."""
    with Image.open(path) as im:
        arr = np.asarray(transforms.Resize((IMG, IMG))(im.convert("RGB")))
    x = torch.from_numpy(arr.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return (x - MEAN) / STD


def denorm(x):
    return (x * STD + MEAN).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()


# ----------------------------------------------------------------------------------------------
# Integrated Gradients and measures
# ----------------------------------------------------------------------------------------------
def ig_positive_map(model, x, target, steps, internal_bs):
    """Positive IG map (224x224): channels summed, negative values set to zero. Zero baseline, Gauss-Legendre."""
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(model)
    x = x.clone().requires_grad_(True)
    attr, delta = ig.attribute(x, baselines=torch.zeros_like(x), target=int(target), n_steps=steps, method="gausslegendre",
                               internal_batch_size=internal_bs, return_convergence_delta=True)
    m = attr.detach().sum(1)[0].clamp(min=0).cpu().numpy()
    return m, float(delta.abs().max())


def pool(m):
    return m.reshape(GRID, CELL, GRID, CELL).mean(axis=(1, 3))


def top_mask(p, frac=0.2):
    thr = np.quantile(p, 1 - frac)
    return p >= thr if thr > 0 else p > 0


def iou_top(p1, p2, frac=0.2):
    a, b = top_mask(p1, frac), top_mask(p2, frac)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def spearman(p1, p2):
    if np.ptp(p1) == 0 or np.ptp(p2) == 0:
        return float("nan")
    return float(stats.spearmanr(p1.ravel(), p2.ravel())[0])


def chance_iou(frac=0.2, n=GRID * GRID, reps=2000, seed=0):
    """Expected IoU of the top-`frac` cells of two unrelated maps (what 'no agreement' looks like)."""
    rng = np.random.default_rng(seed)
    k = int(round(frac * n)); v = []
    for _ in range(reps):
        a = np.zeros(n, bool); b = np.zeros(n, bool)
        a[rng.choice(n, k, replace=False)] = True; b[rng.choice(n, k, replace=False)] = True
        v.append((a & b).sum() / (a | b).sum())
    return float(np.mean(v))


def border_mass(p, w=2):
    """Share of the attribution that falls into the outer `w` cells of the 28x28 grid (uniform: 1-(28-2w)^2/28^2)."""
    t = p.sum()
    if t <= 0:
        return float("nan")
    mk = np.zeros_like(p, bool); mk[:w] = mk[-w:] = True; mk[:, :w] = mk[:, -w:] = True
    return float(p[mk].sum() / t)


def paired_test(d):
    d = np.asarray([x for x in d if not np.isnan(x)])
    if len(d) < 2 or np.all(d == 0):
        return float("nan")
    return float(stats.wilcoxon(d).pvalue)


@torch.no_grad()
def prob_of(model, xs, target, bs=16):
    out = []
    for i in range(0, len(xs), bs):
        out.append(torch.softmax(model(xs[i:i + bs]), 1)[:, target])
    return torch.cat(out).cpu().numpy()


def curves(model, x, pmap, target, n_steps):
    """Deletion / insertion curves in 28x28 cells. Deletion: cells (most relevant first) are replaced by the baseline (0 in
    normalised space); insertion: start from the baseline, put cells back. Returns probability of the explained class
    at fractions 0..1, for the relevance order, and for one fixed random order (reference)."""
    order = np.argsort(-pmap.ravel(), kind="stable")
    rng = np.random.default_rng(0)
    rand = rng.permutation(order.size)
    fracs = np.linspace(0, 1, n_steps + 1)
    res = {}
    for name, ordr in (("rel", order), ("rand", rand)):
        dele, ins = [], []
        for f in fracs:
            k = int(round(f * ordr.size))
            mask = torch.zeros(GRID * GRID)
            mask[torch.from_numpy(ordr[:k].copy())] = 1.0
            mask = mask.view(1, 1, GRID, GRID).repeat_interleave(CELL, 2).repeat_interleave(CELL, 3).to(x.device)
            dele.append(x * (1 - mask))
            ins.append(x * mask)
        res[f"del_{name}"] = prob_of(model, torch.cat(dele), target)
        res[f"ins_{name}"] = prob_of(model, torch.cat(ins), target)
    auc = lambda c: float(np.sum((c[1:] + c[:-1]) / 2.0 * np.diff(fracs)))      # trapezoid rule
    return {k: auc(v) for k, v in res.items()}, fracs


# ----------------------------------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------------------------------
def select_images(args, systems, runs, classes, data):
    """Validation images of seed `args.seed`: seeded random draw per class + all images misclassified by a member of E1."""
    seed = args.seed
    members = ["vim", "kan"]
    per_image = {}                                   # idx -> dict
    for f in ev.FOLDS:
        ref = runs["vim"][(seed, f)]
        probs = {m: runs[m][(seed, f)]["probs"] for m in members}
        e1 = np.mean([probs[m] for m in members], axis=0)
        for j, idx in enumerate(ref["val_idx"]):
            d = dict(index=int(idx), fold=f, label=int(ref["labels"][j]), pos=j)
            for m in members:
                d[f"pred_{m}"] = int(probs[m][j].argmax()); d[f"p_{m}"] = float(probs[m][j].max())
            d["pred_E1"] = int(e1[j].argmax()); d["p_E1"] = float(e1[j].max())
            for m in args.models:
                if m not in members and (seed, f) in runs[m]:
                    pm = runs[m][(seed, f)]["probs"][j]
                    d[f"pred_{m}"] = int(pm.argmax()); d[f"p_{m}"] = float(pm.max())
            per_image[int(idx)] = d
    rng = np.random.default_rng(args.draw_seed)
    chosen = {}
    drawn_order = {}
    for c in range(len(classes)):
        idxs = sorted(i for i, d in per_image.items() if d["label"] == c)
        perm = rng.permutation(len(idxs))
        take = [idxs[k] for k in perm[: args.n_per_class]]
        drawn_order[c] = take
        for i in take:
            chosen[i] = "random_draw"
    wrong = sorted(i for i, d in per_image.items() if d["pred_vim"] != d["label"] or d["pred_kan"] != d["label"] or d["pred_E1"] != d["label"])
    if args.max_misclassified is not None and len(wrong) > args.max_misclassified:
        wrong = [wrong[k] for k in sorted(rng.permutation(len(wrong))[: args.max_misclassified])]
    for i in wrong:
        chosen[i] = "random_draw+misclassified" if i in chosen else "misclassified"
    sel = []
    for i in sorted(chosen):
        d = dict(per_image[i]); d["reason"] = chosen[i]; d["path"] = data.samples[i][0]; d["class"] = classes[d["label"]]
        sel.append(d)
    return sel, drawn_order, wrong


# ----------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------
def overlay(ax, img, m, title, fs=8):
    import matplotlib.pyplot as plt
    ax.imshow(img)
    if m is not None:
        v = np.quantile(m, 0.99)
        a = np.clip(m / v, 0, 1) if v > 0 else m
        ax.imshow(a, cmap="Greens", alpha=0.65, vmin=0, vmax=1)
    ax.set_title(title, fontsize=fs)
    ax.axis("off")


def figure_rows(rows, models, classes, out_png, out_pdf, title_fn, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif"})
    ncol = 1 + len(models)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.1 * ncol, 3.2 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        overlay(axes[r][0], row["img"], None, title_fn(row, None))
        for c, m in enumerate(models, 1):
            overlay(axes[r][c], row["img"], row["maps"][m], title_fn(row, m))
        axes[r][0].set_title(title_fn(row, None), fontsize=8)
    for c, m in enumerate(models, 1):
        axes[0][c].annotate(NICE[m], xy=(0.5, 1.28), xycoords="axes fraction", ha="center", fontsize=10, fontweight="bold")
    axes[0][0].annotate("Original", xy=(0.5, 1.28), xycoords="axes fraction", ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


IMG_IN, HEAD_IN, TITLE_IN, GAP_IN = 2.6, 0.45, 0.33, 0.10          # inches: image side, header, caption above an image, gap


def figure_column(rows, model, title_fn, folder, name, dpi, header):
    """One column of an image figure as its own file (model=None: the original images). Fixed layout (no tight cropping), so
    all columns of one figure have exactly the same size and line up when placed side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif"})
    n = len(rows)
    W, pitch = IMG_IN + 0.2, TITLE_IN + IMG_IN + GAP_IN
    H = HEAD_IN + pitch * n
    fig = plt.figure(figsize=(W, H))
    fig.text(0.5, 1 - HEAD_IN * 0.5 / H, header, ha="center", va="center", fontsize=12, fontweight="bold")
    for r, row in enumerate(rows):
        top = H - HEAD_IN - pitch * r - TITLE_IN
        ax = fig.add_axes([0.1 / W, (top - IMG_IN) / H, IMG_IN / W, IMG_IN / H])
        overlay(ax, row["img"], None if model is None else row["maps"][model], title_fn(row, model), fs=10)
    figio.save_panel(fig, folder, name, dpi, pdf=False, tight=False)
    plt.close(fig)


def export_columns(rows, models, title_fn, folder, prefix, label, dpi):
    for k, m in enumerate([None] + list(models)):
        figure_column(rows, m, title_fn, folder, f"{prefix}{figio.LETTERS[k]}_ig_{label}_{'original' if m is None else SLUG[m]}", dpi,
                      "Original" if m is None else NICE[m])


def draw_metric(ax, which, metrics_rows, models, rand_rows):
    """One measure panel. which: iou (overlap of the maps with the chance level) | ins / del (gain of the relevance order over a random
    order, paired per image) | rand (model-randomisation check)."""
    pairs = [f"{a}_vs_{b}" for a, b in itertools.combinations(models, 2)]
    if which == "iou":
        data = [[r["iou_top20"] for r in metrics_rows if r["kind"] == "pair" and r["model"] == pr and not np.isnan(r["iou_top20"])] for pr in pairs]
        ax.boxplot(data, widths=0.5)
        ax.set_xticklabels([" vs ".join(SHORT[x] for x in pr.split("_vs_")).replace(" vs ", "\nvs\n") for pr in pairs], fontsize=8)
        ax.axhline(chance_iou(), color="k", ls="--", lw=1)
        ax.text(0.5, chance_iou() + 0.004, "chance level", fontsize=8, transform=ax.get_yaxis_transform())
        ax.set_title("Overlap of the top-20% area (IoU)\n(dashed: expected for unrelated maps)", fontsize=10)
    elif which in ("ins", "del"):
        data = []
        for m in models:
            mr = sorted((r for r in metrics_rows if r["kind"] == "model" and r["model"] == m), key=lambda r: r["index"])
            data.append([(r["ins_auc_rel"] - r["ins_auc_rand"]) if which == "ins" else (r["del_auc_rand"] - r["del_auc_rel"]) for r in mr])
        ax.boxplot(data, widths=0.5); ax.set_xticklabels([SHORT[m] for m in models], fontsize=9)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_title("Insertion AUC minus random order\n(> 0: maps find what matters)" if which == "ins" else
                     "Random order minus deletion AUC\n(> 0: maps find what matters)", fontsize=10)
    else:
        vals = [[x for mm, x in rand_rows if mm == m and not np.isnan(x)] for m in models]
        ax.boxplot(vals, widths=0.5); ax.set_xticklabels([SHORT[m] for m in models], fontsize=9)
        ax.axhline(0, color="k", ls="--", lw=1); ax.set_ylim(-1, 1)
        ax.set_title("Model-randomisation check:\nSpearman, trained vs random-init maps (0 = pass)", fontsize=10)
    ax.grid(True, ls=":", alpha=0.6)


METRIC_FILES = (("iou", "overlap_iou"), ("ins", "insertion_gain"), ("del", "deletion_gain"), ("rand", "randomisation_check"))


def figure_metrics(metrics_rows, models, rand_rows, out_png, out_pdf, dpi):
    """The four measure panels side by side (overview file; the manuscript panels are written separately)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif"})
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    for ax, (which, _) in zip(axes, METRIC_FILES):
        draw_metric(ax, which, metrics_rows, models, rand_rows)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def export_metric_panels(metrics_rows, models, rand_rows, folder, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif"})
    for k, (which, nm) in enumerate(METRIC_FILES):
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        draw_metric(ax, which, metrics_rows, models, rand_rows)
        fig.tight_layout()
        figio.save_panel(fig, folder, f"Fig5{figio.LETTERS[k]}_ig_{nm}", dpi, pdf=True)
        plt.close(fig)


# ----------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Integrated Gradients for the epoch-20 models (Ensemble Analysis Plan, section 8)")
    ap.add_argument("--run_dir", default=".")
    ap.add_argument("--data", required=True, help="dataset root (ImageFolder)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--models", nargs="+", default=["vim", "kan"], choices=list(FILES), help="models to explain (E1 members: vim kan)")
    ap.add_argument("--seed", type=int, default=42, help="seed whose validation images and weights are used")
    ap.add_argument("--n_per_class", type=int, default=20)
    ap.add_argument("--max_misclassified", type=int, default=None, help="random cap on the misclassified images (default: all)")
    ap.add_argument("--draw_seed", type=int, default=20261105, help="seed of the random draw (recorded in xai_summary.txt)")
    ap.add_argument("--ig_steps", type=int, default=50)
    ap.add_argument("--ig_batch", type=int, default=10, help="IG internal batch size (lower it if the GPU runs out of memory)")
    ap.add_argument("--curve_steps", type=int, default=20)
    ap.add_argument("--n_random_check", type=int, default=20, help="images for the model-randomisation sanity check")
    ap.add_argument("--n_fig_rows", type=int, default=6, help="rows of the misclassified-images figure")
    ap.add_argument("--per_image", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--paper_dir", default=None, help="folder of the single-panel manuscript figures (default <run_dir>/results/paper_figures); 'none' to skip")
    ap.add_argument("--paper_dpi", type=int, default=figio.PAPER_DPI)
    ap.add_argument("--from_cache", action="store_true", help="skip IG and the measures: re-draw all figures from <out_dir>/xai_cache.pkl of an earlier run (seconds)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    out = a.out_dir or os.path.join(a.run_dir, "results", "xai")
    os.makedirs(out, exist_ok=True)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()
    for m in ("vim", "kan"):
        if m not in a.models:
            a.models = [m] + a.models                    # E1 members are needed for the selection
    a.models = [m for m in ORDER if m in a.models]

    data = datasets.ImageFolder(a.data)
    classes = data.classes
    runs, cl = ev.load_runs(a.run_dir, [a.seed], ev.FOLDS)
    if cl is None or cl != classes:
        raise SystemExit(f"prediction files missing or class names differ from the dataset folders ({cl} vs {classes}).")
    ev.check_consistency(runs, a.run_dir, [a.seed], ev.FOLDS)
    systems = ev.build_systems(runs)
    for m in a.models:
        if len([k for k in runs[m] if k[0] == a.seed]) != 5:
            raise SystemExit(f"{FILES[m]}: stored predictions of seed {a.seed} are incomplete.")

    sel, drawn_order, wrong = select_images(a, systems, runs, classes, data)
    print(f"[*] {len(sel)} images selected: {sum('random_draw' in s['reason'] for s in sel)} from the random draw "
          f"({a.n_per_class} per class), {len(wrong)} misclassified by at least one member of E1 (seed {a.seed}).")

    cache_path = os.path.join(out, "xai_cache.pkl")
    if a.from_cache:
        with open(cache_path, "rb") as fh:
            C = pickle.load(fh)
        if [x["index"] for x in C["sel"]] != [x["index"] for x in sel] or C["models"] != a.models:
            raise SystemExit("the cache belongs to a different selection / model list - run once without --from_cache")
        results, repro, repro_info, rows, rand_rows = C["results"], C["repro"], C["repro_info"], C["rows"], C["rand_rows"]
        for r_ in results.values():                          # stored as float16 to keep the cache small; matplotlib needs float32
            r_["img"] = r_["img"].astype(np.float32)
            r_["maps"] = {m_: v_.astype(np.float32) for m_, v_ in r_["maps"].items()}
        print(f"[*] re-drawing from {cache_path} (no model is loaded)")
    else:
        # ---- explanations, grouped by fold so that each weight file is loaded once -----------------------
        results = {}                                   # idx -> dict(maps={m: 224x224}, targets, probs, img, delta)
        repro = {m: [0, 0] for m in a.models}          # [reproduced stored prediction, total]
        repro_info = {}                                # (model, idx) -> dict(ok, stored_pred, stored_p, new_pred, new_p, delta, rel_delta)
        max_delta = 0.0
        for f in ev.FOLDS:
            todo = [s for s in sel if s["fold"] == f]
            if not todo:
                continue
            for m in a.models:
                w = os.path.join(a.run_dir, "weights", f"{FILES[m]}_v2_seed_{a.seed}_fold_{f}_epoch20.pth")
                if not os.path.isfile(w):
                    raise SystemExit(f"weight file not found: {w}")
                model = build_model(m, len(classes), w).to(device).eval()
                for s in todo:
                    x = load_image(s["path"]).to(device)
                    with torch.no_grad():
                        out_logits = model(x)
                    pred = int(out_logits.argmax(1))
                    repro[m][1] += 1; repro[m][0] += int(pred == s[f"pred_{m}"])
                    mp, delta = ig_positive_map(model, x, pred, a.ig_steps, a.ig_batch)
                    max_delta = max(max_delta, delta)
                    with torch.no_grad():
                        gap = float(out_logits[0, pred] - model(torch.zeros_like(x))[0, pred])      # f(x) - f(baseline), logit units
                    repro_info[(m, s["index"])] = dict(ok=int(pred == s[f"pred_{m}"]), stored_pred=s[f"pred_{m}"], stored_p=s[f"p_{m}"],
                                                       new_pred=pred, new_p=float(torch.softmax(out_logits, 1)[0, pred]),
                                                       delta=delta, rel_delta=delta / (abs(gap) + 1e-8))
                    r = results.setdefault(s["index"], dict(maps={}, pooled={}, target={}, prob={}, curves={}, img=denorm(x.cpu())))
                    r["maps"][m] = mp; r["pooled"][m] = pool(mp); r["target"][m] = pred
                    r["prob"][m] = float(torch.softmax(out_logits, 1)[0, pred])
                    r["curves"][m], _ = curves(model, x, r["pooled"][m], pred, a.curve_steps)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            print(f"    fold {f}: {len(todo)} images done ({time.time() - t0:.0f}s)")

        # ---- measures ----------------------------------------------------------------------------------------
        rows = []
        for s in sel:
            r = results[s["index"]]
            for m in a.models:
                c = r["curves"][m]
                ri = repro_info[(m, s["index"])]
                rows.append(dict(kind="model", index=s["index"], model=m, target=r["target"][m], prob=r["prob"][m],
                                 del_auc_rel=c["del_rel"], del_auc_rand=c["del_rand"], ins_auc_rel=c["ins_rel"], ins_auc_rand=c["ins_rand"],
                                 repro_ok=ri["ok"], ig_delta=ri["delta"], ig_rel_delta=ri["rel_delta"], border_mass=border_mass(r["pooled"][m]),
                                 correct=int(r["target"][m] == s["label"])))
            for m1, m2 in itertools.combinations(a.models, 2):
                p1, p2 = r["pooled"][m1], r["pooled"][m2]
                rows.append(dict(kind="pair", index=s["index"], model=f"{m1}_vs_{m2}", same_target=int(r["target"][m1] == r["target"][m2]),
                                 iou_top20=iou_top(p1, p2), spearman=spearman(p1, p2),
                                 spearman_raw_pixels=spearman(r["maps"][m1], r["maps"][m2])))
        # ---- model-randomisation sanity check ------------------------------------------------------------------
        rand_rows = []
        check = [s for s in sel][: a.n_random_check]
        for m in a.models:
            model = build_model(m, len(classes), None, seed=12345).to(device).eval()
            for s in check:
                x = load_image(s["path"]).to(device)
                with torch.no_grad():
                    pred = int(model(x).argmax(1))
                mp, _ = ig_positive_map(model, x, pred, a.ig_steps, a.ig_batch)
                rand_rows.append((m, spearman(results[s["index"]]["pooled"][m], pool(mp))))
            del model

        slim = {i_: dict(r_, maps={m_: v_.astype(np.float16) for m_, v_ in r_["maps"].items()}, img=r_["img"].astype(np.float16)) for i_, r_ in results.items()}
        with open(cache_path, "wb") as fh:
            pickle.dump(dict(sel=sel, models=a.models, results=slim, repro=repro, repro_info=repro_info, rows=rows, rand_rows=rand_rows), fh, protocol=4)

    # ---- outputs -------------------------------------------------------------------------------------------
    with open(os.path.join(out, "xai_selection.csv"), "w", newline="", encoding="utf-8") as fh:
        keys = ["index", "path", "class", "label", "fold", "reason"] + [f"pred_{m}" for m in a.models] + [f"p_{m}" for m in a.models] + ["pred_E1", "p_E1"]
        w = csv.writer(fh); w.writerow(keys)
        for s in sel:
            w.writerow([s.get(k, "") for k in keys])
    with open(os.path.join(out, "xai_metrics.csv"), "w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("kind", "index", "model"), k))
        w = csv.writer(fh); w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k, "") for k in keys])
    np.savez_compressed(os.path.join(out, "xai_maps_pooled.npz"), **{f"{m}_{i}": results[i]["pooled"][m] for i in results for m in a.models})

    def label(c):
        return classes[c]

    def title_fn(row, m):
        s = row["sel"]
        if m is None:
            return f"{s['class']} (true) | fold {s['fold']}"
        return f"{label(row['target'][m])}  p = {row['prob'][m]:.2f}"

    def make_row(i):
        s = next(x for x in sel if x["index"] == i); r = results[i]
        return dict(sel=s, img=r["img"], maps=r["maps"], target=r["target"], prob=r["prob"])

    first = [drawn_order[c][0] for c in range(len(classes)) if drawn_order[c]]
    figure_rows([make_row(i) for i in first], a.models, classes, os.path.join(out, "Figure_IG_classes.png"),
                os.path.join(out, "Figure_IG_classes.pdf"), title_fn, a.dpi)
    conf = [c for c in range(len(classes)) if classes[c] in ("Medium1", "Medium2", "Walnut1", "Walnut2") and drawn_order[c]]
    if conf:                                         # compact main-text figure: first image of the seeded draw of each confusable class
        figure_rows([make_row(drawn_order[c][0]) for c in conf], a.models, classes, os.path.join(out, "Figure_IG_main.png"),
                    os.path.join(out, "Figure_IG_main.pdf"), title_fn, a.dpi)
    if wrong:
        pick = list(np.random.default_rng(a.draw_seed + 1).permutation(wrong)[: a.n_fig_rows])

        def title_wrong(row, m):
            s = row["sel"]
            if m is None:
                return f"true {s['class']} | E1: {label(s['pred_E1'])}"
            return f"{label(row['target'][m])}  p = {row['prob'][m]:.2f}"
        figure_rows([make_row(int(i)) for i in pick], a.models, classes, os.path.join(out, "Figure_IG_misclassified.png"),
                    os.path.join(out, "Figure_IG_misclassified.pdf"), title_wrong, a.dpi)
    paper_dir = a.paper_dir or os.path.join(a.run_dir, "results", "paper_figures")
    if paper_dir.lower() != "none":
        art, sup = figio.paper_dirs(paper_dir)
        print("\nPaper panels:")
        if conf:
            export_columns([make_row(drawn_order[c][0]) for c in conf], a.models, title_fn, art, "Fig4", "main", a.paper_dpi)
        export_metric_panels(rows, a.models, rand_rows, art, a.paper_dpi)
        export_columns([make_row(i) for i in first], a.models, title_fn, sup, "FigS2", "classes", a.paper_dpi)
        if wrong:
            export_columns([make_row(int(i)) for i in pick], a.models, title_wrong, sup, "FigS3", "misclassified", a.paper_dpi)
    figure_metrics(rows, a.models, rand_rows, os.path.join(out, "Figure_IG_metrics.png"), os.path.join(out, "Figure_IG_metrics.pdf"), a.dpi)
    if a.per_image:
        os.makedirs(os.path.join(out, "per_image"), exist_ok=True)
        for s in sel:
            figure_rows([make_row(s["index"])], a.models, classes, os.path.join(out, "per_image", f"IG_{s['class']}_{s['index']}.png"), None,
                        title_fn, 150)

    mean = lambda v: float(np.nanmean(v)) if len(v) else float("nan")
    chance = chance_iou()
    ok_all = {s_["index"] for s_ in sel if all(repro_info[(m, s_["index"])]["ok"] for m in a.models)}
    lines = [f"Integrated Gradients, seed {a.seed}, epoch-20 weights of the fold in which each image is a validation image",
             f"Selection: {len(sel)} images = seeded random draw ({a.n_per_class} per class, draw seed {a.draw_seed}) + {len(wrong)} images misclassified by a member of E1",
             f"IG: target = predicted class logit, {a.ig_steps} Gauss-Legendre steps, zero baseline (normalised space), positive attributions",
             f"Comparison grid: {GRID}x{GRID} cells of {CELL}x{CELL} px; top-20% overlap = IoU of the 20% highest cells (chance level {chance:.3f}); curves with {a.curve_steps} steps", ""]
    lines.append("Integrity check - stored epoch-20 prediction reproduced from the weight file + the image file:")
    for m in a.models:
        lines.append(f"  {NICE[m]}: {repro[m][0]}/{repro[m][1]}" + ("" if repro[m][0] == repro[m][1] else "   <-- MISMATCH"))
        for (mm, idx), ri in sorted(repro_info.items()):
            if mm == m and not ri["ok"]:
                lines.append(f"      image index {idx}: stored class {classes[ri['stored_pred']]} (p={ri['stored_p']:.3f}) vs recomputed {classes[ri['new_pred']]} (p={ri['new_p']:.3f})")
    lines.append(f"  images reproduced by every model: {len(ok_all)}/{len(sel)} (statistics below are given for all images and for this subset)")
    lines.append("")
    lines.append("IG completeness (|convergence delta|; relative = delta / |f(x) - f(baseline)| of the explained logit):")
    for m in a.models:
        mr = [r for r in rows if r["kind"] == "model" and r["model"] == m]
        d = np.array([r["ig_delta"] for r in mr]); rd = np.array([r["ig_rel_delta"] for r in mr])
        lines.append(f"  {NICE[m]}: delta median {np.median(d):.3f}, max {d.max():.3f}; relative median {np.median(rd):.3f}, 90th pct {np.quantile(rd, .9):.3f}, "
                     f"images with relative delta > 0.1: {int((rd > 0.1).sum())}/{len(rd)}")
    lines.append("")
    lines.append("Agreement of the maps between models (per image, then mean):")
    for m1, m2 in itertools.combinations(a.models, 2):
        pr = [r for r in rows if r["kind"] == "pair" and r["model"] == f"{m1}_vs_{m2}"]
        lines.append(f"  {SHORT[m1]} vs {SHORT[m2]}: top-20% IoU {mean([r['iou_top20'] for r in pr]):.3f} (chance {chance:.3f})   "
                     f"Spearman (28x28) {mean([r['spearman'] for r in pr]):.3f}   Spearman (raw pixels) {mean([r['spearman_raw_pixels'] for r in pr]):.3f}   "
                     f"same explained class {sum(r['same_target'] for r in pr)}/{len(pr)}")
    lines.append("")
    lines.append("Insertion / deletion AUC (mean over images; the random order is the reference for 'no information'):")
    gains = {}
    for m in a.models:
        mr = [r for r in rows if r["kind"] == "model" and r["model"] == m]
        lines.append(f"  {NICE[m]}: deletion {mean([r['del_auc_rel'] for r in mr]):.3f} (random {mean([r['del_auc_rand'] for r in mr]):.3f}) | "
                     f"insertion {mean([r['ins_auc_rel'] for r in mr]):.3f} (random {mean([r['ins_auc_rand'] for r in mr]):.3f})")
        gains[m] = {r["index"]: (r["ins_auc_rel"] - r["ins_auc_rand"], r["del_auc_rand"] - r["del_auc_rel"], r["correct"]) for r in mr}
    for tag, keep in (("all images", None), ("images reproduced by every model", ok_all)):
        lines += ["", f"Gain of the relevance order over the random order, {tag} (insertion gain / deletion gain; share of images with gain > 0; Wilcoxon p vs 0):"]
        for m in a.models:
            g = [v for i_, v in gains[m].items() if keep is None or i_ in keep]
            ins = np.array([v[0] for v in g]); de = np.array([v[1] for v in g])
            lines.append(f"  {SHORT[m]} (n={len(g)}): insertion {ins.mean():+.3f} ({(ins > 0).mean():.0%}, p={paired_test(ins):.2g}) | "
                         f"deletion {de.mean():+.3f} ({(de > 0).mean():.0%}, p={paired_test(de):.2g})")
        for m1, m2 in itertools.combinations(a.models, 2):
            ids = [i_ for i_ in gains[m1] if keep is None or i_ in keep]
            di = [gains[m1][i_][0] - gains[m2][i_][0] for i_ in ids]; dd = [gains[m1][i_][1] - gains[m2][i_][1] for i_ in ids]
            lines.append(f"  {SHORT[m1]} minus {SHORT[m2]}: insertion gain {np.mean(di):+.3f} (p={paired_test(di):.2g}), deletion gain {np.mean(dd):+.3f} (p={paired_test(dd):.2g})")
    lines += ["", "Attribution mass in the outer two cell rows/columns of the 28x28 grid (uniform expectation "
              f"{1 - (GRID - 4) ** 2 / GRID ** 2:.3f}; large values = border / patch-grid artefact):"]
    for m in a.models:
        lines.append(f"  {SHORT[m]}: {mean([r['border_mass'] for r in rows if r['kind'] == 'model' and r['model'] == m]):.3f}")
    lines += ["", f"Model-randomisation sanity check (Spearman between the maps of the trained and of a randomly initialised model, "
                  f"first {a.n_random_check} selected images; values near 0 mean the maps depend on the learned weights):"]
    for m in a.models:
        v = [x for mm, x in rand_rows if mm == m]
        lines.append(f"  {NICE[m]}: mean {mean(v):.3f}, median {float(np.nanmedian(v)):.3f} over {len(v)} images")
    text = "\n".join(lines)
    with open(os.path.join(out, "xai_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\n" + text + f"\n\nFiles written to {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
