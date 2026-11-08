"""
==============================================================================
Author: Prof. Dr. Metin Zontul
Project: XAI-VisionMamba-KAN-Ensemble
File: secondary_train.py

Secondary (reference) models trained with EXACTLY the protocol of master_train.py:
    resnet50      ImageNet-1k V2 ResNet-50, single Linear(2048 -> 6) head
    vit_b16       ImageNet-1k V1 ViT-B/16 (torchvision), LayerNorm -> Linear(768 -> 6) head
    vit_b16_mlp   same ViT-B/16 backbone, LayerNorm -> Linear(768,256) - SiLU - Dropout(0.3) - Linear(256,6)
                  (the SAME MLP head as the ResNet-Mamba-MLP hybrid, hidden 256)
    vit_b16_kan   same ViT-B/16 backbone, LayerNorm -> real B-spline KAN(768 -> 6) head
                  (G = 5, r = 3, grid [-3, 3]; the SAME KANLayer class as the hybrid)

Everything else is imported from master_train.py, so the following are IDENTICAL to the
main study: 3 seeds (42, 1024, 2026) x stratified 5-fold CV with identical splits,
20 epochs, constant LR, epoch-20 weights only, focal loss (gamma 2), AdamW lr 1e-4 /
wd 1e-2, batch 32, fp32 + grad-clip 1.0, k*90-degree rotations, seed rule
seed*100 + fold, per-fold npz/json/weights/confusion matrices, resume, failure guard.

Why a LayerNorm in front of the ViT heads?
    The hybrid's KAN head receives a LayerNorm-ed feature vector and its spline grid
    covers [-3, 3]. To keep the KAN-vs-linear comparison on ViT clean, BOTH ViT heads
    receive the same parameter-free LayerNorm; only the head itself differs.

Output files use the same naming scheme and the same folders as master_train.py
(weights/, preds/, results/, logs/), so ensemble_eval.py / make_table1.py can read them.
Summary / model-card files get the suffix "_secondary" and never overwrite the main ones.

Typical use
-----------
  python secondary_train.py --data ".../Marble" --out_dir . --benchmark   # time estimate
  python secondary_train.py --data ".../Marble" --out_dir .               # all four
  python secondary_train.py --models resnet50 vit_b16                      # subset
  (interrupted runs resume; finished folds are skipped)
==============================================================================
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16

import mamba_core
import master_train as mt
from mamba_core import count_parameters
from resnet_mamba_kan import KANLayer


class ResNet50Linear(nn.Module):
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        self.pretrained = pretrained
        self.pretrained_source = "ImageNet-1k V2 (torchvision ResNet50_Weights.IMAGENET1K_V2)" if pretrained else "none"

    def forward(self, x):
        return self.net(x)

    def describe(self) -> Dict:
        total, trainable = count_parameters(self)
        return {"model": "ResNet-50, linear head", "head": "Linear(2048, 6)",
                "head_parameters": sum(p.numel() for p in self.net.fc.parameters()),
                "parameters_total": total, "parameters_trainable": trainable}


class ViTB16(nn.Module):
    """ViT-B/16 (all 12 pre-trained encoder blocks untouched); only the classification head differs."""

    def __init__(self, head: str, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        assert head in ("linear", "mlp", "kan")
        self.head_type = head
        self.net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        d = self.net.heads.head.in_features                              # 768
        norm = nn.LayerNorm(d, elementwise_affine=False)                 # parameter-free, same for both heads
        if head == "linear":
            last = nn.Linear(d, num_classes)
        elif head == "mlp":                                              # identical to ResNetMambaMLP.mlp_head except the input width
            last = nn.Sequential(nn.Linear(d, 256), nn.SiLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))
        else:
            last = KANLayer(d, num_classes, grid_size=5, spline_order=3)
        self.net.heads = nn.Sequential(norm, last)
        self.pretrained = pretrained
        self.pretrained_source = "ImageNet-1k V1 (torchvision ViT_B_16_Weights.IMAGENET1K_V1)" if pretrained else "none"

    def forward(self, x):
        return self.net(x)

    def describe(self) -> Dict:
        total, trainable = count_parameters(self)
        head = ("Linear(768, 6)" if self.head_type == "linear"
                else "Linear(768,256) - SiLU - Dropout(0.3) - Linear(256,6)" if self.head_type == "mlp"
                else "KAN 768 -> 6 (single layer), G=5, r=3, grid range [-3, 3], 8 B-spline bases + SiLU base path per edge")
        return {"model": f"ViT-B/16, {self.head_type} head",
                "head": "parameter-free LayerNorm(768) -> " + head,
                "head_parameters": sum(p.numel() for p in self.net.heads.parameters()),
                "parameters_total": total, "parameters_trainable": trainable}


def start_logging(log_dir: str) -> str:
    """Like master_train.start_logging, but the file is called SECONDARY_TRAINING_LOG_<time>.txt."""
    from datetime import datetime
    path = os.path.join(log_dir, f"SECONDARY_TRAINING_LOG_{datetime.now():%Y%m%d_%H%M%S}.txt")
    fh = open(path, "x", encoding="utf-8")
    sys.stdout, sys.stderr = mt.Tee(sys.__stdout__, fh), mt.Tee(sys.__stderr__, fh)
    return path


def build_specs(args, num_classes):
    specs = [
        dict(name="resnet50", build=lambda pretrained: ResNet50Linear(num_classes, pretrained), seed_offset=0),
        dict(name="vit_b16", build=lambda pretrained: ViTB16("linear", num_classes, pretrained), seed_offset=0),
        dict(name="vit_b16_mlp", build=lambda pretrained: ViTB16("mlp", num_classes, pretrained), seed_offset=0),
        dict(name="vit_b16_kan", build=lambda pretrained: ViTB16("kan", num_classes, pretrained), seed_offset=0),
    ]
    if args.models:
        unknown = set(args.models) - {s["name"] for s in specs}
        if unknown:
            raise SystemExit(f"Unknown model(s): {sorted(unknown)} (choose from resnet50 vit_b16 vit_b16_mlp vit_b16_kan)")
        specs = [s for s in specs if s["name"] in args.models]
    return specs


def main(argv=None):
    args = mt.parse_args(argv)
    dirs = {k: os.path.join(args.out_dir, k) for k in ("weights", "preds", "results", "logs")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    # Summaries of this script go to their own sub-folder so that the main summary is never overwritten.
    sum_dirs = dict(dirs, results=os.path.join(dirs["results"], "secondary_summary"))
    os.makedirs(sum_dirs["results"], exist_ok=True)
    log_path = start_logging(dirs["logs"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    if args.tf32 and device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"[*] SECONDARY TRAINING STARTED. DEVICE: {device}. Log file: {log_path}")

    if not os.path.isdir(args.data):
        print(f"[-] ERROR: dataset folder not found: {args.data}")
        return 1
    print("[*] Loading and resizing the dataset once (kept in memory) ...")
    data = mt.MarbleData(args.data)
    groups = mt.load_groups(args.groups_csv, data.paths, args.data) if args.groups_csv else None
    # print_environment mentions the Mamba back-end; harmless here (these models do not use it)
    mamba_core.set_scan_backend("auto")
    mt.print_environment(args, device, data, groups)
    num_classes = len(data.classes)
    specs = build_specs(args, num_classes)

    if args.benchmark:
        mt.benchmark(specs, data, args, device)
        return 0

    cards = {}
    for spec in specs:                                                    # pre-flight: downloads the pre-trained weights now
        m = spec["build"](not args.no_pretrained)
        cards[spec["name"]] = dict(m.describe(), pretrained_weights=getattr(m, "pretrained_source", "none"))
        print(f"[MODEL CARD] {spec['name']}: {json.dumps(cards[spec['name']], indent=2)}")
        del m
    with open(os.path.join(dirs["results"], f"model_cards_{mt.PROTOCOL_TAG}_secondary.json"), "w", encoding="utf-8") as fh:
        json.dump(cards, fh, indent=2)

    start = time.time()
    rows = []
    for seed in args.seeds:
        print(f"\n{'-' * 20} SEED: {seed} {'-' * 20}")
        splits = mt.make_splits(data.targets, groups, seed, args.folds)
        for fold, (tr, va) in enumerate(splits, 1):
            for spec in specs:
                print(f"\n{'=' * 60}\nSTARTING: {spec['name']} | seed {seed} | fold {fold}/{args.folds} "
                      f"| train {len(tr)} / val {len(va)}\n{'=' * 60}")
                rows.append(mt.run_fold(spec, seed, fold, tr, va, data, args, dirs, device))
                mt.write_csv_and_summary(rows, data.classes, sum_dirs, args.epochs, echo=False)
    mt.write_csv_and_summary(rows, data.classes, sum_dirs, args.epochs)
    print(f"\n[*] ALL SECONDARY TRAININGS COMPLETED. Total Duration: {(time.time() - start) / 60:.2f} minutes.")
    print(f"[*] Complete screen log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
