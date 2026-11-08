"""
Author: Prof. Dr. Metin Zontul
Project: XAI-VisionMamba-KAN-Ensemble
File: mamba_core.py

Shared selective state-space (Mamba) building blocks used by ALL Mamba-based
models in this project (Vision Mamba, ResNet-Mamba-MLP, ResNet-Mamba-KAN).

What is implemented here
------------------------
* MambaMixer   : the bidirectional Mamba mixer of Vision Mamba (Zhu et al.,
                 ICML 2024; "bimamba_type = v2", output divided by 2). Parameter
                 names, shapes and initialisation follow the official Vim /
                 Mamba code (A initialised as S4D-real, dt_rank = ceil(d_model/16),
                 dt bias initialised in [1e-3, 1e-1], separate forward / backward
                 conv1d, x_proj, dt_proj, A, D; shared in_proj and out_proj).
                 => the official Vim checkpoints can be loaded without renaming.
* VimBlock     : Add -> RMSNorm -> bidirectional Mamba mixer (pre-norm residual).
* selective scan: the recurrence
                       h_k = exp(dt_k * A) * h_{k-1} + dt_k * B_k * x_k      (Eq. 6)
                       y_k = C_k . h_k + D * x_k                              (Eq. 7)
                 followed by the z-gate  y * SiLU(z).
                 Two interchangeable back-ends compute exactly this recurrence:
                   "cuda"  : official mamba_ssm CUDA kernel (needs Linux / WSL2)
                   "torch" : pure PyTorch parallel (Hillis-Steele) scan, no
                             compilation needed, works on Windows.
                 The back-end only changes speed, not the model.
* ResNetMambaBackbone: ResNet-50 (ImageNet, torchvision IMAGENET1K_V2 = the weights
                 `resnet50-11ad3fa6.pth` used in the earlier experiments) feature extractor
                 -> 7x7=49 tokens of 2048 channels -> token-wise linear projection to 512
                 -> ONE VimBlock -> final RMSNorm -> mean pooling -> LayerNorm. Both hybrid models use this very
                 same backbone; they differ only in the classification head.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50, ResNet50_Weights

try:  # official CUDA kernel (optional)
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_selective_scan
except Exception:  # ImportError, OSError (missing .so), ...
    _cuda_selective_scan = None

# ---------------------------------------------------------------------------
# Selective scan (Eq. 6 and Eq. 7)
# ---------------------------------------------------------------------------
_SCAN_BACKEND = "auto"  # "auto" | "torch" | "cuda"


def set_scan_backend(name: str) -> None:
    """Choose the scan implementation. 'auto' uses the CUDA kernel if available."""
    global _SCAN_BACKEND
    if name not in ("auto", "torch", "cuda"):
        raise ValueError(f"Unknown scan backend: {name}")
    if name == "cuda" and _cuda_selective_scan is None:
        raise RuntimeError("scan backend 'cuda' requested but mamba_ssm is not importable.")
    _SCAN_BACKEND = name


def active_scan_backend(on_cuda: bool = True) -> str:
    """Name of the back-end that will actually be used."""
    if _SCAN_BACKEND == "torch":
        return "torch"
    if _cuda_selective_scan is not None and on_cuda:
        return "cuda"
    return "torch"


def selective_scan_parallel(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    """
    Pure-PyTorch selective scan with the same signature as
    mamba_ssm.ops.selective_scan_interface.selective_scan_fn.

        u, delta, z : (batch, d_inner, L)
        A           : (d_inner, N)  (negative real)
        B, C        : (batch, N, L)
        D           : (d_inner,)
        delta_bias  : (d_inner,)

    The linear recurrence h_k = a_k * h_{k-1} + b_k is evaluated with an
    inclusive Hillis-Steele scan over the associative operator
        (a1, b1) o (a2, b2) = (a1*a2, a2*b1 + b2)
    which needs ceil(log2 L) vectorised steps instead of L sequential ones.
    All a_k = exp(dt_k*A) lie in (0, 1], so the scan is numerically stable.
    The computation is always carried out in float32.
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    A = A.float()
    B = B.float()
    C = C.float()

    a = torch.exp(delta.unsqueeze(2) * A[None, :, :, None])          # (b, d, n, L)
    b = (delta * u).unsqueeze(2) * B.unsqueeze(1)                    # (b, d, n, L)

    L = u.shape[-1]
    off = 1
    while off < L:
        b = torch.cat([b[..., :off], a[..., off:] * b[..., :-off] + b[..., off:]], dim=-1)
        if off * 2 < L:  # `a` is not needed after the last step
            a = torch.cat([a[..., :off], a[..., off:] * a[..., :-off]], dim=-1)
        off *= 2

    y = torch.einsum("bdnl,bnl->bdl", b, C)                          # Eq. 7: y = C . h
    if D is not None:
        y = y + u * D.float()[None, :, None]
    if z is not None:
        y = y * F.silu(z.float())
    return y.to(dtype_in)


def selective_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    if active_scan_backend(u.is_cuda) == "cuda":
        return _cuda_selective_scan(
            u.contiguous(), delta.contiguous(), A.float(), B.contiguous(), C.contiguous(),
            D.float() if D is not None else None, z=z.contiguous() if z is not None else None,
            delta_bias=delta_bias.float() if delta_bias is not None else None,
            delta_softplus=delta_softplus,
        )
    return selective_scan_parallel(u, delta, A, B, C, D, z, delta_bias, delta_softplus)


# ---------------------------------------------------------------------------
# Normalisation, stochastic depth
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """RMSNorm with a single weight vector (same parameter layout as mamba_ssm.RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)


def drop_path(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if p == 0.0 or not training:
        return x
    keep = 1.0 - p
    mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
    return x * mask / keep


# ---------------------------------------------------------------------------
# Bidirectional Mamba mixer (Vision Mamba, bimamba v2)
# ---------------------------------------------------------------------------
class MambaMixer(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, conv_bias=True):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # forward branch
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # backward branch
        self.conv1d_b = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
                                  padding=d_conv - 1, bias=conv_bias)
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        for proj in (self.dt_proj, self.dt_proj_b):
            nn.init.uniform_(proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                           + math.log(dt_min)).clamp(min=dt_init_floor)
            with torch.no_grad():
                proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
            proj.bias._no_reinit = True

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)  # S4D-real
        self.A_log = nn.Parameter(torch.log(A))
        self.A_b_log = nn.Parameter(torch.log(A.clone()))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D_b = nn.Parameter(torch.ones(self.d_inner))

    def _branch(self, xz, conv, x_proj, dt_proj, A_log, D):
        L = xz.shape[-1]
        x, z = xz.chunk(2, dim=1)                                        # (b, d_inner, L) each
        x = F.silu(conv(x)[..., :L])                                     # causal depth-wise conv
        x_dbl = x_proj(x.transpose(1, 2))                                # (b, L, dt_rank + 2N)
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = (dt @ dt_proj.weight.t()).transpose(1, 2)                   # (b, d_inner, L); bias added in the scan
        return selective_scan(
            x, dt, -torch.exp(A_log.float()), Bm.transpose(1, 2).contiguous(), Cm.transpose(1, 2).contiguous(),
            D.float(), z=z, delta_bias=dt_proj.bias.float(), delta_softplus=True)

    def forward(self, hidden_states):                                    # (b, L, d_model)
        xz = self.in_proj(hidden_states).transpose(1, 2)                 # (b, 2*d_inner, L)
        out_f = self._branch(xz, self.conv1d, self.x_proj, self.dt_proj, self.A_log, self.D)
        out_b = self._branch(xz.flip([-1]), self.conv1d_b, self.x_proj_b, self.dt_proj_b, self.A_b_log, self.D_b)
        out = (out_f + out_b.flip([-1])).transpose(1, 2) / 2             # "div2" variant of Vim
        return self.out_proj(out)


def init_mamba_weights(module: nn.Module, n_layer: int) -> None:
    """Initialisation of the official Mamba/Vim code (zero Linear biases except dt bias,
    out_proj rescaled by 1/sqrt(n_layer))."""
    for m in module.modules():
        if isinstance(m, nn.Linear) and m.bias is not None and not getattr(m.bias, "_no_reinit", False):
            nn.init.zeros_(m.bias)
        if isinstance(m, MambaMixer):
            nn.init.kaiming_uniform_(m.out_proj.weight, a=math.sqrt(5))
            with torch.no_grad():
                m.out_proj.weight /= math.sqrt(n_layer)


class VimBlock(nn.Module):
    """Add -> RMSNorm -> bidirectional Mamba mixer. Returns (mixer_out, residual)."""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, drop_path_rate=0.0, norm_eps=1e-5,
                 residual_in_fp32=True):
        super().__init__()
        self.mixer = MambaMixer(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        self.residual_in_fp32 = residual_in_fp32

    def forward(self, hidden_states, residual: Optional[torch.Tensor] = None):
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + drop_path(hidden_states, self.drop_path_rate, self.training)
        hidden_states = self.norm(residual.to(self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        return self.mixer(hidden_states), residual


def run_block(layer: nn.Module, hidden, residual, use_checkpoint: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_checkpoint and layer.training and torch.is_grad_enabled():
        return checkpoint(layer, hidden, residual, use_reentrant=False)
    return layer(hidden, residual)


# ---------------------------------------------------------------------------
# Shared ResNet-50 + Mamba backbone of the two hybrid models
# ---------------------------------------------------------------------------
class ResNetMambaBackbone(nn.Module):
    """
    image (3x224x224) -> ResNet-50 conv trunk -> (2048, 7, 7) -> 49 tokens x 2048
                      -> token-wise Linear(2048 -> d_model=512)
                      -> VimBlock (bidirectional Mamba, pre-norm, residual)
                      -> final RMSNorm -> mean over tokens -> LayerNorm -> (512,)
    The projection keeps the Mamba block at d_model = 512 (d_inner = 1024) instead of
    d_model = 2048 (4x more parameters and a 4x larger scan state).
    """
    CNN_DIM = 2048

    def __init__(self, pretrained: bool = True, d_model: int = 512, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, drop_path_rate: float = 0.0):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])          # drops avg-pool + fc
        self.proj = nn.Linear(self.CNN_DIM, d_model)
        self.block = VimBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                              drop_path_rate=drop_path_rate)
        self.norm_f = RMSNorm(d_model)
        self.pool_norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        init_mamba_weights(self.block, n_layer=1)

    def forward(self, x):
        feats = self.cnn(x)                                              # (b, 2048, 7, 7)
        seq = self.proj(feats.flatten(2).transpose(1, 2))                # (b, 49, 512)
        hidden, residual = self.block(seq, None)
        hidden = self.norm_f(residual + drop_path(hidden, self.block.drop_path_rate, self.training))
        return self.pool_norm(hidden.mean(dim=1))                        # (b, 512)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
