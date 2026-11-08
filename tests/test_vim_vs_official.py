"""Full-model check: our VisionMamba == OFFICIAL hustvl/Vim VisionMamba (fetched from GitHub),
with only the CUDA-only kernels replaced by their PyTorch reference equivalents in the official code path.
Verifies parameter names/shapes (=> official checkpoints load) and forward outputs."""
import sys, types, importlib.util, torch, torch.nn as nn, torch.nn.functional as F, math, warnings
warnings.filterwarnings("ignore")
import os as _os; sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))

def mod(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m
def load(name, path, g=None):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m

# ---- stubs for CUDA-only deps ---------------------------------------------------------------
for n in ("selective_scan_cuda", "causal_conv1d_cuda"): mod(n)
mod("causal_conv1d", causal_conv1d_fn=None, causal_conv1d_update=None)
ssi = load("ssi_ref", _os.path.join(_os.environ.get("VIMSRC", "/tmp/vimsrc"), "vim_ssi.py"))                     # official reference scan + autograd fns
def mamba_inner_fn_no_out_proj(xz, conv_w, conv_b, x_proj_w, dt_proj_w, A, B=None, C=None, D=None,
                               delta_bias=None, B_proj_bias=None, C_proj_bias=None, delta_softplus=True):
    # == MambaInnerFnNoOutProj.forward with the CUDA kernels replaced by the official reference ops
    L = xz.shape[-1]; delta_rank = dt_proj_w.shape[1]; d_state = A.shape[-1]
    x, z = xz.chunk(2, dim=1)
    x = F.silu(F.conv1d(x, conv_w, conv_b, padding=conv_w.shape[-1] - 1, groups=conv_w.shape[0])[..., :L])
    x_dbl = F.linear(x.transpose(1, 2).reshape(-1, x.shape[1]), x_proj_w)
    delta = (dt_proj_w @ x_dbl[:, :delta_rank].t()).reshape(-1, xz.shape[0], L).permute(1, 0, 2)
    Bv = x_dbl[:, delta_rank:delta_rank + d_state].reshape(xz.shape[0], L, d_state).transpose(1, 2).contiguous()
    Cv = x_dbl[:, -d_state:].reshape(xz.shape[0], L, d_state).transpose(1, 2).contiguous()
    return ssi.selective_scan_ref(x, delta, A, Bv, Cv, D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus)
mod("mamba_ssm"); mod("mamba_ssm.ops")
mod("mamba_ssm.ops.selective_scan_interface", selective_scan_fn=ssi.selective_scan_ref,
    mamba_inner_fn=None, bimamba_inner_fn=None, mamba_inner_fn_no_out_proj=mamba_inner_fn_no_out_proj)
mod("mamba_ssm.ops.triton"); 
class RMSNorm(nn.Module):                                             # mamba_ssm.ops.triton.layer_norm.RMSNorm layout
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(hidden_size)); self.register_parameter("bias", None)
def rms_norm_fn(x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-6, **kw):
    if residual is not None:
        x = x.float() + residual.float() if residual_in_fp32 else x + residual
    out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    out = out.to(weight.dtype)
    return (out, x if not residual_in_fp32 else x.float()) if prenorm else out
mod("mamba_ssm.ops.triton.layer_norm", RMSNorm=RMSNorm, rms_norm_fn=rms_norm_fn, layer_norm_fn=None)
mod("mamba_ssm.modules")
load("mamba_ssm.modules.mamba_simple", _os.path.join(_os.environ.get("VIMSRC", "/tmp/vimsrc"), "vim_mamba_simple.py"))
mod("mamba_ssm.utils"); mod("mamba_ssm.utils.generation", GenerationMixin=object)
mod("mamba_ssm.utils.hf", load_config_hf=None, load_state_dict_hf=None)
mod("rope")
import timm.layers as tl
mod("timm.models.registry", register_model=lambda f: f)
mod("timm.models.layers", trunc_normal_=tl.trunc_normal_, lecun_normal_=tl.lecun_normal_, DropPath=tl.DropPath, to_2tuple=tl.to_2tuple)
import timm.models.vision_transformer as tv
if not hasattr(tv, "_load_weights"): tv._load_weights = None
official = load("official_models_mamba", _os.path.join(_os.environ.get("VIMSRC", "/tmp/vimsrc"), "models_mamba.py"))

# ---- compare ---------------------------------------------------------------------------------
import vision_mamba_train as ours
for variant, fn in (("tiny", official.vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2),
                    ("small", official.vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2)):
    torch.manual_seed(0)
    off = fn(num_classes=6).eval()
    mine = ours.create_vision_mamba(6, variant=variant, pretrained=False).eval()
    sd_o, sd_m = off.state_dict(), mine.state_dict()
    assert set(sd_o) == set(sd_m), (sorted(set(sd_o) ^ set(sd_m))[:10])
    assert all(sd_o[k].shape == sd_m[k].shape for k in sd_o), "shape mismatch"
    n_o = sum(p.numel() for p in off.parameters()); n_m = sum(p.numel() for p in mine.parameters())
    print(f"[{variant}] keys identical ({len(sd_o)}), params official={n_o:,} ours={n_m:,}")
    assert n_o == n_m
    # randomise so that the check is not trivially passed by near-zero weights
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for k, v in sd_o.items():
            if v.dtype.is_floating_point and not k.endswith(("A_log", "A_b_log")):
                v.add_(0.02 * torch.randn(v.shape, generator=g))
    mine.load_state_dict(sd_o, strict=True)                      # official state dict loads into our model
    x = torch.randn(2, 3, 224, 224, generator=g)
    with torch.no_grad():
        yo, ym = off(x), mine(x)
        fo, fm = off.forward_features(x), mine.forward_features(x)
    print(f"[{variant}] max|logits diff|={(yo-ym).abs().max():.2e}  max|features diff|={(fo-fm).abs().max():.2e}  (|logit| ~ {yo.abs().mean():.3f})")
    assert torch.allclose(yo, ym, atol=1e-4, rtol=1e-3) and torch.allclose(fo, fm, atol=1e-4, rtol=1e-3)
print("FULL VIM MODEL == OFFICIAL IMPLEMENTATION")
