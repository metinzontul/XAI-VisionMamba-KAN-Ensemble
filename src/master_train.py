"""
==============================================================================
Author: Prof. Dr. Metin Zontul
Project: XAI-VisionMamba-KAN-Ensemble
File: master_train.py

Master training script for the three Mamba-based models of the study
(Vision Mamba, ResNet-Mamba-MLP, ResNet-Mamba-KAN).

Protocol (answers to the reviewers' requests)
---------------------------------------------
 1. FIXED SCHEDULE, FINAL CHECKPOINT. Every model is trained for exactly 20 epochs
    with a constant learning rate; the weights after epoch 20 are saved and are the
    ONLY weights ever evaluated. There is no best-epoch / best-checkpoint selection and
    no early stopping, so the validation fold is never used to select anything.
 2. 15 EVALUATIONS PER MODEL = 3 seeds (42, 1024, 2026) x stratified 5-fold CV. The split
    depends only on (seed, labels[, groups]) and is therefore IDENTICAL for all models
    (paired comparisons, ensembles of models validated on the same images).
 3. The same hyper-parameters, augmentation, optimiser and schedule for all models. Loss, weight decay, batch size,
    learning rate, optimiser and the augmentation family are those of the earlier experiments (Table 1: focal loss gamma=2,
    AdamW lr 1e-4 / wd 1e-2, batch 32, RandomResizedCrop + flips + rotation) so that the new models are comparable with them.
    Differences to the earlier pipeline: epoch-20 weights instead of best-epoch, fp32 + grad-clip 1.0 instead of AMP,
    rotation by k*90 deg instead of zero-filled random-angle rotation.
 4. For every (model, seed, fold) the script stores: epoch-20 weights, the validation
    class-probabilities / logits / labels / image indices (npz; needed for the ensemble,
    McNemar and Nadeau-Bengio tests), accuracy, macro-F1, per-class recall and the
    confusion matrix (json + csv).  A model card (architecture, pre-training, optimiser,
    parameter count, KAN grid/order) is printed and saved.
 5. The complete screen output is also written, unedited, to a time-stamped log file.
 6. Numerical safety: fp32 training, gradient clipping, and a non-finite-loss guard. A
    diverged run is REPORTED as failed (never silently counted).
 7. Optional group-wise split (--groups_csv) if tile / slab identities are known.

Typical use
-----------
  python master_train.py --benchmark                 # speed / memory / run-time estimate
  python master_train.py                             # full study (3 models x 15 runs)
  python master_train.py --with_vim_control          # + a 2nd Vim with another init seed
  (an interrupted run continues where it stopped: finished folds are skipped)
==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import mamba_core

PROTOCOL_TAG = "v2"          # part of every output file name; the old runs can never be overwritten
SEEDS = [42, 1024, 2026]
IMG_SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ----------------------------------------------------------------------------
# Logging: everything printed (stdout AND stderr) also goes to a time-stamped file
# ----------------------------------------------------------------------------
class Tee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, message):
        self.stream.write(message)
        self.fh.write(message)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


def start_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"MASTER_TRAINING_LOG_{datetime.now():%Y%m%d_%H%M%S}.txt")
    fh = open(path, "x", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, fh), Tee(sys.__stderr__, fh)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Data: images are decoded + resized ONCE and kept in memory as uint8 tensors
# ----------------------------------------------------------------------------
class MarbleData:
    def __init__(self, root: str):
        base = datasets.ImageFolder(root)
        self.root, self.classes = root, base.classes
        self.targets = np.array(base.targets)
        self.paths = [p for p, _ in base.samples]
        resize = transforms.Resize((IMG_SIZE, IMG_SIZE))               # same resize as the earlier experiments
        imgs = []
        for p in self.paths:
            with Image.open(p) as im:
                imgs.append(torch.from_numpy(np.asarray(resize(im.convert("RGB")))).permute(2, 0, 1).contiguous())
        self.images = torch.stack(imgs)                                 # (N, 3, 224, 224) uint8
        rel = sorted(f"{os.path.relpath(p, root).replace(os.sep, '/')}|{os.path.getsize(p)}" for p in self.paths)   # same value on Windows and Linux
        self.fingerprint = hashlib.sha256("\n".join(rel).encode()).hexdigest()[:16]

    def class_counts(self):
        return {c: int((self.targets == i).sum()) for i, c in enumerate(self.classes)}


class FocalLoss(nn.Module):
    """Focal loss as in the earlier experiments (alpha = 1, gamma = 2, mean reduction)."""
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        return ((1.0 - torch.exp(-ce)) ** self.gamma * ce).mean()


class Rot90:
    """Random rotation by a multiple of 90 degrees (no interpolation, NO black corners)."""
    def __call__(self, x):
        k = int(torch.randint(0, 4, (1,)))
        return torch.rot90(x, k, dims=(1, 2)) if k else x


# Training-only augmentation (validation images are only resized + normalised).
# Same family as the earlier experiments (RandomResizedCrop(224, scale 0.8-1), h/v flip, rotation) with ONE change:
# the zero-filled random-angle RandomRotation(90) is replaced by rotations of k*90 deg, which create no black corners.
TRAIN_AUG = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    Rot90(),
])
AUG_DESCRIPTION = "train only: RandomResizedCrop(224, scale 0.8-1.0), random h-flip, v-flip, rotation by k*90 deg (no zero fill)"


class FoldDataset(Dataset):
    def __init__(self, data: MarbleData, indices, train: bool):
        self.data, self.indices, self.train = data, np.asarray(indices), train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = self.data.images[idx]
        if self.train:
            x = TRAIN_AUG(x)
        x = (x.float() / 255.0 - MEAN) / STD
        return x, int(self.data.targets[idx])


def load_groups(csv_path: str, paths, root: str):
    """CSV with header `filename,group` (relative path under the dataset root OR bare file name)."""
    table = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            table[row["filename"].replace("\\", "/")] = row["group"]
    groups = []
    for p in paths:
        rel = os.path.relpath(p, root).replace("\\", "/")
        key = rel if rel in table else os.path.basename(p)
        if key not in table:
            raise KeyError(f"No group given for image {rel} in {csv_path}")
        groups.append(table[key])
    _, codes = np.unique(groups, return_inverse=True)
    return codes


def make_splits(targets, groups, seed: int, k: int):
    if groups is None:   # identical to the splits of the earlier experiments
        it = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(targets)), targets)
    else:                # tiles / slabs never straddle training and validation
        it = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(targets)), targets, groups)
    return [(np.sort(tr), np.sort(va)) for tr, va in it]


# ----------------------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------------------
def build_specs(args, num_classes):
    from vision_mamba_train import create_vision_mamba, resolve_pretrained_path
    from resnet_mamba_mlp import ResNetMambaMLP
    from resnet_mamba_kan import ResNetMambaKAN

    cache = {}

    def vim(pretrained):
        ckpt = None
        if pretrained:
            if "p" not in cache:
                cache["p"] = resolve_pretrained_path(args.vim_variant, args.vim_ckpt)
            ckpt = cache["p"]
        return create_vision_mamba(num_classes, variant=args.vim_variant, pretrained=pretrained, ckpt_path=ckpt)

    specs = [
        dict(name="vision_mamba", build=lambda pretrained: vim(pretrained), seed_offset=0),
        dict(name="resnet_mamba_mlp", build=lambda pretrained: ResNetMambaMLP(num_classes, pretrained=pretrained), seed_offset=0),
        dict(name="resnet_mamba_kan", build=lambda pretrained: ResNetMambaKAN(num_classes, pretrained=pretrained), seed_offset=0),
    ]
    if args.with_vim_control:   # same splits, different initialisation / data order / augmentation seed
        specs.append(dict(name="vision_mamba_ctrl", build=lambda pretrained: vim(pretrained), seed_offset=7919))
    if args.models:
        unknown = set(args.models) - {s["name"] for s in specs}
        if unknown:
            raise SystemExit(f"Unknown model(s): {sorted(unknown)}")
        specs = [s for s in specs if s["name"] in args.models]
    return specs


# ----------------------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, logits_all, labels_all = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        logits_all.append(out.float().cpu())
        labels_all.append(y.cpu())
    logits, labels = torch.cat(logits_all), torch.cat(labels_all)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("non-finite validation logits")
    return dict(loss=loss_sum / len(labels), acc=(logits.argmax(1) == labels).float().mean().item() * 100,
                logits=logits, probs=torch.softmax(logits, 1), labels=labels)


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss ({loss.item()})")
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(gnorm):
            raise FloatingPointError(f"non-finite gradient norm ({gnorm.item()})")
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total * 100


def fold_metrics(res, class_names):
    y, p = res["labels"].numpy(), res["logits"].argmax(1).numpy()
    labels = list(range(len(class_names)))
    rec = recall_score(y, p, labels=labels, average=None, zero_division=0)
    return dict(
        val_acc=res["acc"], val_loss=res["loss"],
        macro_f1=float(f1_score(y, p, labels=labels, average="macro", zero_division=0) * 100),
        recall_per_class={c: float(r * 100) for c, r in zip(class_names, rec)},
        confusion_matrix=confusion_matrix(y, p, labels=labels).tolist(), n_val=int(len(y)))


def run_fold(spec, seed, fold, train_idx, val_idx, data, args, dirs, device):
    E = args.epochs
    tag = f"{spec['name']}_{PROTOCOL_TAG}_seed_{seed}_fold_{fold}"
    w_path = os.path.join(dirs["weights"], f"{tag}_epoch{E}.pth")
    p_path = os.path.join(dirs["preds"], f"{tag}.npz")
    j_path = os.path.join(dirs["results"], f"{tag}.json")

    if args.resume and os.path.isfile(j_path) and os.path.isfile(w_path) and os.path.isfile(p_path):
        with open(j_path, encoding="utf-8") as fh:
            old = json.load(fh)
        if old.get("status") == "ok":
            print(f"-> [{spec['name']}] seed {seed} fold {fold}: already completed "
                  f"(Final Epoch {E} Val Acc {old['val_acc']:.2f}%) - skipped (resume).")
            return old

    base = seed * 100 + fold + spec["seed_offset"]
    set_seed(base)
    model = spec["build"](not args.no_pretrained).to(device)
    criterion = FocalLoss(args.focal_gamma) if args.loss == "focal" else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator().manual_seed(base)
    kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(FoldDataset(data, train_idx, True), batch_size=args.batch_size, shuffle=True, generator=gen, **kw)
    val_loader = DataLoader(FoldDataset(data, val_idx, False), batch_size=32, shuffle=False, **kw)

    record = dict(model=spec["name"], seed=seed, fold=fold, epochs=E, protocol=PROTOCOL_TAG)
    t0 = time.time()
    try:
        for epoch in range(1, E + 1):
            te = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
            res = evaluate(model, val_loader, criterion, device)
            print(f"Epoch [{epoch}/{E}] - Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.2f}% | "
                  f"Val Loss: {res['loss']:.4f}, Val Acc: {res['acc']:.2f}% [{time.time() - te:.1f}s]")
    except FloatingPointError as e:
        record.update(status="failed", error=str(e), epoch_reached=epoch)
        with open(j_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        print(f"!!! [{spec['name']}] seed {seed} fold {fold} FAILED at epoch {epoch}: {e}. "
              f"This run is NOT saved as a result and is listed as failed in the summary.")
        return record

    m = fold_metrics(res, data.classes)
    os.makedirs(dirs["weights"], exist_ok=True)
    torch.save(model.state_dict(), w_path)                              # epoch-E weights, nothing else is saved
    np.savez_compressed(p_path, probs=res["probs"].numpy().astype(np.float32), logits=res["logits"].numpy().astype(np.float32),
                        labels=res["labels"].numpy(), val_idx=np.asarray(val_idx), class_names=np.array(data.classes),
                        model=spec["name"], seed=seed, fold=fold, epoch=E)

    # Re-load the saved file and make sure it reproduces the logged predictions.
    verified = None
    if args.verify_checkpoint:
        chk = spec["build"](False).to(device)
        chk.load_state_dict(torch.load(w_path, map_location=device))
        again = evaluate(chk, val_loader, criterion, device)
        diff = (again["probs"] - res["probs"]).abs().max().item()
        verified = bool(torch.equal(again["probs"].argmax(1), res["probs"].argmax(1)) and diff < 1e-3)
        if not verified:
            print(f"!!! checkpoint re-load check FAILED for {tag} (max prob diff {diff:.2e})")
        del chk

    record.update(status="ok", train_loss=tr_loss, train_acc=tr_acc, seconds=time.time() - t0,
                  checkpoint_verified=verified, weights=w_path, preds=p_path, **m)
    with open(j_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(f"-> Fold {fold} Completed. Final (Epoch {E}) Validation Accuracy: {m['val_acc']:.2f}%. Weights saved: {w_path}")
    print(f"   macro-F1 {m['macro_f1']:.2f}% | recall per class: "
          + ", ".join(f"{c} {r:.1f}" for c, r in m["recall_per_class"].items()))
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def write_csv_and_summary(rows, class_names, dirs, epochs, echo=True):
    path = os.path.join(dirs["results"], f"fold_metrics_{PROTOCOL_TAG}.csv")
    cols = ["model", "seed", "fold", "status", "val_acc", "macro_f1", "val_loss", "train_acc", "train_loss", "n_val"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols + [f"recall_{c}" for c in class_names] + ["weights", "preds"])
        for r in sorted(rows, key=lambda r: (r["model"], r["seed"], r["fold"])):
            ok = r.get("status") == "ok"
            w.writerow([r.get(c, "") for c in cols] +
                       [f"{r['recall_per_class'][c]:.4f}" if ok else "" for c in class_names] +
                       [r.get("weights", ""), r.get("preds", "")])
    lines = [f"\n{'=' * 78}", f"SUMMARY - FINAL (EPOCH {epochs}) WEIGHTS, no checkpoint selection. mean +- sample SD (n-1) over seed-fold runs",
             f"{'=' * 78}",
             f"{'model':22s} {'n_ok':>5s} {'Val Acc %':>16s} {'min':>7s} {'max':>7s} {'macro-F1 %':>16s}"]
    models = sorted({r["model"] for r in rows})
    failed = []
    for mname in models:
        ok = [r for r in rows if r["model"] == mname and r.get("status") == "ok"]
        failed += [r for r in rows if r["model"] == mname and r.get("status") != "ok"]
        if not ok:
            lines.append(f"{mname:22s} {0:5d}  (no completed run)")
            continue
        a = np.array([r["val_acc"] for r in ok]); f = np.array([r["macro_f1"] for r in ok])
        sd = lambda v: v.std(ddof=1) if len(v) > 1 else float("nan")
        lines.append(f"{mname:22s} {len(ok):5d} {a.mean():9.2f} +- {sd(a):4.2f} {a.min():7.2f} {a.max():7.2f} {f.mean():9.2f} +- {sd(f):4.2f}")
    if failed:
        lines.append("\nFAILED RUNS (diverged - NOT included above; report them or re-train with another protocol-conform setting):")
        lines += [f"  {r['model']} seed {r['seed']} fold {r['fold']}: {r.get('error')}" for r in failed]
    lines.append(f"\nPer-fold table: {path}")
    text = "\n".join(lines)
    if echo:
        print(text)
    with open(os.path.join(dirs["results"], f"summary_{PROTOCOL_TAG}.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


def print_environment(args, device, data, groups):
    print(f"[*] Started {datetime.now():%Y-%m-%d %H:%M:%S} | python {platform.python_version()} | {platform.platform()}")
    print(f"[*] torch {torch.__version__} | torchvision {torchvision.__version__} | numpy {np.__version__}")
    if device.type == "cuda":
        print(f"[*] GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
    print(f"[*] Mamba selective-scan back-end: {mamba_core.active_scan_backend(device.type == 'cuda')} "
          f"(official CUDA kernel available: {mamba_core._cuda_selective_scan is not None})")
    print(f"[*] Arguments: {json.dumps(vars(args), sort_keys=True)}")
    print(f"[*] Dataset: {data.root} | {len(data.targets)} images | fingerprint {data.fingerprint}")
    print(f"[*] Class counts: {json.dumps(data.class_counts())}")
    print(f"[*] Split: " + ("StratifiedKFold, image-level (NO tile/slab independence established)" if groups is None
                           else f"StratifiedGroupKFold over {len(np.unique(groups))} groups"))
    print(f"[*] Optimiser AdamW lr={args.lr} wd={args.weight_decay} | const. LR | grad-clip {args.grad_clip} | "
          f"batch {args.batch_size} | epochs {args.epochs} | loss {args.loss}{' gamma=' + str(args.focal_gamma) if args.loss == 'focal' else ''} | "
          f"fp32{' (TF32 matmul)' if args.tf32 else ''} | final-epoch weights only")
    print(f"[*] Augmentation ({AUG_DESCRIPTION}); validation: resize 224x224 + ImageNet normalisation")


def benchmark(specs, data, args, device):
    n_train = int(len(data.targets) * (1 - 1 / args.folds)); n_val = len(data.targets) - n_train
    total_h = 0.0
    for spec in specs:
        torch.manual_seed(0)
        model = spec["build"](False).to(device)
        opt = optim.AdamW(model.parameters(), lr=args.lr)
        x = torch.randn(args.batch_size, 3, IMG_SIZE, IMG_SIZE, device=device)
        y = torch.randint(0, len(data.classes), (args.batch_size,), device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        times = []
        for i in range(4):                                              # 1 warm-up + 3 timed
            t = time.time(); model.train(); opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward(); opt.step()
            if device.type == "cuda": torch.cuda.synchronize()
            if i: times.append(time.time() - t)
        it = float(np.mean(times))
        model.eval()
        with torch.no_grad():
            t = time.time(); model(x)
            if device.type == "cuda": torch.cuda.synchronize()
            vit = time.time() - t
        epoch_s = it * math.ceil(n_train / args.batch_size) + vit * math.ceil(n_val / args.batch_size)
        hours = epoch_s * args.epochs * len(args.seeds) * args.folds / 3600
        total_h += hours
        mem = f", peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if device.type == "cuda" else ""
        print(f"[benchmark] {spec['name']:20s} {it:6.2f} s/train-iter -> ~{epoch_s:6.1f} s/epoch -> "
              f"~{hours:5.1f} h for {len(args.seeds) * args.folds} runs{mem} | params {model.describe()['parameters_total']:,}")
        del model, opt
    print(f"[benchmark] ESTIMATED TOTAL (data loading excluded): ~{total_h:.1f} h")


# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Master training script (Vim, ResNet-Mamba-MLP, ResNet-Mamba-KAN)")
    p.add_argument("--data", default="../../dataset/Marble")
    p.add_argument("--out_dir", default=".", help="weights/, preds/, results/, logs/ are created below this directory")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--loss", choices=["focal", "ce"], default="focal", help="focal = loss of the earlier experiments (Table 1)")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--models", nargs="+", default=None, help="subset of: vision_mamba resnet_mamba_mlp resnet_mamba_kan vision_mamba_ctrl")
    p.add_argument("--vim_variant", choices=["tiny", "small"], default="small")
    p.add_argument("--vim_ckpt", default=None, help="local official Vim .pth (otherwise downloaded from Hugging Face)")
    p.add_argument("--no_pretrained", action="store_true", help="random initialisation (tests only; NOT the study protocol)")
    p.add_argument("--with_vim_control", action="store_true", help="also train a 2nd Vim (other init seed) for the 'two-Vim' ensemble control")
    p.add_argument("--groups_csv", default=None, help="CSV (filename,group) with tile/slab ids -> group-wise folds")
    p.add_argument("--scan_backend", choices=["auto", "torch", "cuda"], default="auto")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--no_resume", dest="resume", action="store_false", help="retrain even if a fold is already finished")
    p.add_argument("--no_verify_checkpoint", dest="verify_checkpoint", action="store_false")
    p.add_argument("--no_tf32", dest="tf32", action="store_false")
    p.add_argument("--benchmark", action="store_true", help="time a few iterations per model, print the projected total run-time, exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dirs = {k: os.path.join(args.out_dir, k) for k in ("weights", "preds", "results", "logs")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    log_path = start_logging(dirs["logs"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    if args.tf32 and device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    mamba_core.set_scan_backend(args.scan_backend)
    print(f"[*] TRAINING STARTED. DEVICE: {device}. Log file: {log_path}")

    if not os.path.isdir(args.data):
        print(f"[-] ERROR: dataset folder not found: {args.data}")
        return 1
    print("[*] Loading and resizing the dataset once (kept in memory) ...")
    data = MarbleData(args.data)
    groups = load_groups(args.groups_csv, data.paths, args.data) if args.groups_csv else None
    print_environment(args, device, data, groups)
    num_classes = len(data.classes)
    specs = build_specs(args, num_classes)

    if args.benchmark:
        benchmark(specs, data, args, device)
        return 0

    # Pre-flight: build every model once (downloads pre-trained weights NOW, not hours into the run) and save the model cards.
    cards = {}
    for spec in specs:
        m = spec["build"](not args.no_pretrained)
        cards[spec["name"]] = dict(m.describe(), pretrained_weights=getattr(m, "pretrained_source", None) or
                                   ("ImageNet-1k V2 (ResNet-50 trunk)" if not args.no_pretrained else "none"))
        print(f"[MODEL CARD] {spec['name']}: {json.dumps(cards[spec['name']], indent=2)}")
        del m
    with open(os.path.join(dirs["results"], f"model_cards_{PROTOCOL_TAG}.json"), "w", encoding="utf-8") as fh:
        json.dump(cards, fh, indent=2)

    start = time.time()
    rows = []
    for seed in args.seeds:
        print(f"\n{'-' * 20} SEED: {seed} {'-' * 20}")
        splits = make_splits(data.targets, groups, seed, args.folds)
        with open(os.path.join(dirs["results"], f"splits_seed_{seed}.json"), "w", encoding="utf-8") as fh:
            json.dump({"seed": seed, "fingerprint": data.fingerprint, "val_idx": [v.tolist() for _, v in splits]}, fh)
        for fold, (tr, va) in enumerate(splits, 1):                      # fold-major: complete model sets per fold
            for spec in specs:
                print(f"\n{'=' * 60}\nSTARTING: {spec['name']} | seed {seed} | fold {fold}/{args.folds} "
                      f"| train {len(tr)} / val {len(va)}\n{'=' * 60}")
                rows.append(run_fold(spec, seed, fold, tr, va, data, args, dirs, device))
                write_csv_and_summary(rows, data.classes, dirs, args.epochs, echo=False)   # crash-safe partial table
    write_csv_and_summary(rows, data.classes, dirs, args.epochs)
    print(f"\n[*] ALL TRAININGS COMPLETED. Total Duration: {(time.time() - start) / 60:.2f} minutes.")
    print(f"[*] Complete screen log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
