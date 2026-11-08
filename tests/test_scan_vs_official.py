"""Checks mamba_core against the OFFICIAL Mamba/Vim reference code (fetched from GitHub into /tmp/vimsrc).
1) parallel scan == official sequential selective_scan_ref (forward + gradients)
2) MambaMixer (v2, div2) == official Mamba mixer math
"""
import sys, types, importlib.util, torch, torch.nn.functional as F
import os as _os; sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import mamba_core as mc

# --- load the official reference scan (needs einops -> tiny shim) -------------------------------
import re
src = open(_os.path.join(_os.environ.get("VIMSRC", "/tmp/vimsrc"), "vim_ssi.py")).read()
einops = types.ModuleType("einops")
def rearrange(t, pat, **kw):
    if pat == "d -> d 1": return t[:, None]
    raise NotImplementedError(pat)
def repeat(*a, **k): raise NotImplementedError
einops.rearrange, einops.repeat = rearrange, repeat
sys.modules["einops"] = einops
for name in ("selective_scan_cuda", "causal_conv1d_cuda"):
    sys.modules[name] = types.ModuleType(name)
spec = importlib.util.spec_from_loader("ref", loader=None); ref = importlib.util.module_from_spec(spec)
try:
    exec(compile(src, "vim_ssi.py", "exec"), ref.__dict__)
except Exception as e:
    print("note: ref import issue:", repr(e)); raise
selective_scan_ref = ref.__dict__["selective_scan_ref"]

torch.manual_seed(0)
def rand_inputs(b=2, d=12, n=16, L=197, dtype=torch.float32):
    u = torch.randn(b, d, L, dtype=dtype, requires_grad=True)
    delta = torch.randn(b, d, L, dtype=dtype, requires_grad=True)
    A = (-torch.exp(torch.randn(d, n))).requires_grad_(True)
    B = torch.randn(b, n, L, requires_grad=True); C = torch.randn(b, n, L, requires_grad=True)
    D = torch.randn(d, requires_grad=True); z = torch.randn(b, d, L, requires_grad=True)
    bias = torch.randn(d, requires_grad=True)
    return [u, delta, A, B, C, D, z, bias]

for L in (1, 2, 5, 49, 64, 197):
    ins = rand_inputs(L=L)
    out_ref = selective_scan_ref(ins[0], ins[1], ins[2], ins[3], ins[4], ins[5], ins[6], ins[7], True)
    g_ref = [t if t is not None else torch.zeros_like(i) for t,i in zip(torch.autograd.grad(out_ref.pow(2).sum(), ins, allow_unused=True), ins)]
    out = mc.selective_scan_parallel(*ins[:7], delta_bias=ins[7], delta_softplus=True)
    g = [t if t is not None else torch.zeros_like(i) for t,i in zip(torch.autograd.grad(out.pow(2).sum(), ins, allow_unused=True), ins)]
    err = (out - out_ref).abs().max().item()
    gerr = max(((a - b).abs().max() / (b.abs().max() + 1e-9)).item() for a, b in zip(g, g_ref))
    print(f"scan L={L:3d}  max|out-ref|={err:.2e}  max rel grad err={gerr:.2e}")
    assert err < 1e-4 and gerr < 1e-3, "scan mismatch"

# --- mixer vs official math written exactly as in mamba_simple (non-fused, v2) -------------------
mix = mc.MambaMixer(48, d_state=16)
x = torch.randn(3, 20, 48)
def official_branch(xz, conv, x_proj, dt_proj, A_log, D):
    L = xz.shape[-1]; xx, z = xz.chunk(2, dim=1)
    xx = F.silu(conv(xx)[..., :L])
    x_dbl = x_proj(xx.transpose(1, 2).reshape(-1, xx.shape[1]))              # (bl, d)
    dt, B, C = torch.split(x_dbl, [mix.dt_rank, 16, 16], dim=-1)
    dt = dt_proj.weight @ dt.t()                                             # d (b l)
    dt = dt.reshape(dt.shape[0], xz.shape[0], L).permute(1, 0, 2)            # b d l
    B = B.reshape(xz.shape[0], L, 16).transpose(1, 2).contiguous(); C = C.reshape(xz.shape[0], L, 16).transpose(1, 2).contiguous()
    return selective_scan_ref(xx, dt, -torch.exp(A_log.float()), B, C, D.float(), z=z, delta_bias=dt_proj.bias.float(), delta_softplus=True)
xz = (x @ mix.in_proj.weight.t()).transpose(1, 2)
o = official_branch(xz, mix.conv1d, mix.x_proj, mix.dt_proj, mix.A_log, mix.D)
ob = official_branch(xz.flip([-1]), mix.conv1d_b, mix.x_proj_b, mix.dt_proj_b, mix.A_b_log, mix.D_b)
want = F.linear((o + ob.flip([-1])).transpose(1, 2) / 2, mix.out_proj.weight)
got = mix(x)
print("mixer max|diff| vs official math:", (got - want).abs().max().item())
assert torch.allclose(got, want, atol=1e-5)
print("ALL SCAN/MIXER CHECKS PASSED")
