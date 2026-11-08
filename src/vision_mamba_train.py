"""
Author: Prof. Dr. Metin Zontul
Project: XAI-VisionMamba-KAN-Ensemble
File: vision_mamba_train.py

Vision Mamba (Vim) of Zhu et al., "Vision Mamba: Efficient Visual Representation
Learning with Bidirectional State Space Model" (ICML 2024).

This is the *complete* Vim architecture, not a transformer with a Mamba block on top:

    image -> 16x16 patch embedding -> class token inserted in the MIDDLE of the
    sequence -> + absolute position embedding -> 24 x [Add -> RMSNorm -> bidirectional
    selective-SSM mixer (bimamba v2, output / 2)] -> Add + RMSNorm -> class-token
    feature -> linear classifier.

It is the official `vim_{tiny,small}_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_
with_midclstok_div2` configuration. No self-attention layer exists in the network.
Parameter names are identical to the official implementation, so the official
ImageNet-1k pre-trained Vim checkpoints (Hugging Face: hustvl/Vim-small-midclstok,
hustvl/Vim-tiny-midclstok) load directly; only the 1000-class head is replaced.

The selective scan runs either on the official CUDA kernel (if `mamba_ssm` is installed,
Linux/WSL2) or on the pure-PyTorch parallel scan of `mamba_core.py` (Windows). Both
evaluate the same recurrence (Eq. 6 and Eq. 7 of the manuscript).
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from mamba_core import (RMSNorm, VimBlock, active_scan_backend, count_parameters, drop_path,
                        init_mamba_weights, run_block)

VIM_VARIANTS = {
    # name: (embed_dim, depth, Hugging Face repo with ImageNet-1k weights)
    "tiny": (192, 24, "hustvl/Vim-tiny-midclstok"),
    "small": (384, 24, "hustvl/Vim-small-midclstok"),
}


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)                  # (b, N, d)


class VisionMamba(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=384, depth=24, d_state=16, num_classes=1000,
                 drop_path_rate=0.1, norm_eps=1e-5, use_checkpoint="auto"):
        super().__init__()
        self.embed_dim, self.depth, self.d_state = embed_dim, depth, d_state
        self.use_checkpoint = use_checkpoint
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.head = nn.Linear(embed_dim, num_classes)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr                                         # as in the official code
        self.layers = nn.ModuleList([
            VimBlock(embed_dim, d_state=d_state, drop_path_rate=inter_dpr[i], norm_eps=norm_eps)
            for i in range(depth)])
        self.norm_f = RMSNorm(embed_dim, eps=norm_eps)
        self.drop_path_rate = drop_path_rate

        # --- initialisation as in the official implementation ---
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        init_mamba_weights(self, n_layer=depth)

    # ------------------------------------------------------------------
    def forward_features(self, x):
        x = self.patch_embed(x)
        b, m, _ = x.shape
        pos = m // 2                                                    # class token in the middle
        x = torch.cat((x[:, :pos], self.cls_token.expand(b, -1, -1), x[:, pos:]), dim=1)
        x = x + self.pos_embed

        use_ckpt = self.use_checkpoint
        if use_ckpt == "auto":                                          # only the PyTorch scan needs it
            use_ckpt = active_scan_backend(x.is_cuda) == "torch"
        residual, hidden = None, x
        for layer in self.layers:
            hidden, residual = run_block(layer, hidden, residual, use_ckpt)
        residual = residual + drop_path(hidden, self.drop_path_rate, self.training)
        hidden = self.norm_f(residual.to(self.norm_f.weight.dtype))
        return hidden[:, pos, :]

    def forward(self, x):
        return self.head(self.forward_features(x))

    # ------------------------------------------------------------------
    def describe(self) -> Dict:
        total, trainable = count_parameters(self)
        return {
            "model": f"Vision Mamba (Vim) embed_dim={self.embed_dim}, depth={self.depth}",
            "reference": "Zhu et al., ICML 2024 (bimamba v2, middle class token, abs. pos. embed, div2)",
            "self_attention_layers": 0,
            "patch_size": 16, "d_state": self.d_state, "d_conv": 4, "expand": 2,
            "dt_rank": self.layers[0].mixer.dt_rank,
            "drop_path_rate": self.drop_path_rate,
            "scan_backend": active_scan_backend(torch.cuda.is_available()),
            "parameters_total": total, "parameters_trainable": trainable,
        }


# ----------------------------------------------------------------------
# Pre-trained weights
# ----------------------------------------------------------------------
def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Official checkpoints also pickle an argparse.Namespace -> needs weights_only=False.
        print(f"[!] weights_only=True failed for {path}; loading with weights_only=False "
              f"(checkpoint comes from the official hustvl/Vim release).")
        return torch.load(path, map_location="cpu", weights_only=False)


def resolve_pretrained_path(variant: str, ckpt_path: Optional[str] = None) -> str:
    if ckpt_path:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Vim checkpoint not found: {ckpt_path}")
        return ckpt_path
    from huggingface_hub import hf_hub_download, list_repo_files  # lazy import
    repo = VIM_VARIANTS[variant][2]
    files = [f for f in list_repo_files(repo) if f.endswith(".pth")]
    # The repositories also hold a long-sequence fine-tuned "..._ft_..." checkpoint; the study uses the
    # standard ImageNet-1k pre-trained model (224x224, 16x16 patches), i.e. the file WITHOUT "_ft_".
    plain = [f for f in files if "_ft_" not in f]
    if len(plain) != 1:
        raise RuntimeError(f"Expected exactly one non-fine-tuned .pth in {repo}, found {files}. "
                           f"Download the right one manually and pass --vim_ckpt <file>.")
    print(f"[*] Vim pre-trained checkpoint: {repo}/{plain[0]} (other files in the repo ignored: {[f for f in files if f != plain[0]]})")
    return hf_hub_download(repo, plain[0])


def load_pretrained_vim(model: VisionMamba, variant: str, ckpt_path: Optional[str] = None) -> str:
    path = resolve_pretrained_path(variant, ckpt_path)
    ckpt = _torch_load(path)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    state = {k: v for k, v in state.items() if not k.startswith("head.")}   # 1000-class head is dropped
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"head.weight", "head.bias"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"Vim checkpoint does not match the model.\n  missing: {sorted(set(missing) - allowed_missing)[:8]}"
                           f"\n  unexpected: {sorted(unexpected)[:8]}")
    return path


def create_vision_mamba(num_classes: int = 6, variant: str = "small", pretrained: bool = True,
                        ckpt_path: Optional[str] = None, drop_path_rate: float = 0.1) -> VisionMamba:
    embed_dim, depth, _ = VIM_VARIANTS[variant]
    model = VisionMamba(embed_dim=embed_dim, depth=depth, num_classes=1000, drop_path_rate=drop_path_rate)
    source = "random initialisation"
    if pretrained:
        source = f"ImageNet-1k Vim-{variant} ({load_pretrained_vim(model, variant, ckpt_path)})"
    model.head = nn.Linear(embed_dim, num_classes)                      # new task head (official init)
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)
    model.pretrained_source = source
    return model
