"""
Run in WSL2/Linux after `pip install mamba-ssm`: compares the official CUDA selective-scan kernel with the
pure-PyTorch scan on a full Vim-S forward/backward pass (same weights, same input).
    python check_cuda_backend.py
"""
import torch
import mamba_core
from vision_mamba_train import create_vision_mamba

assert torch.cuda.is_available(), "no CUDA device"
assert mamba_core._cuda_selective_scan is not None, "mamba_ssm is not importable"
torch.manual_seed(0)
model = create_vision_mamba(6, "small", pretrained=False).cuda().train()
x = torch.randn(4, 3, 224, 224, device="cuda")
res = {}
for backend in ("torch", "cuda"):
    mamba_core.set_scan_backend(backend)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1)                       # same drop-path masks
    out = model(x)
    out.sum().backward()
    g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    res[backend] = (out.detach(), g)
    print(f"{backend:5s}: logits[0,:3] = {out[0, :3].tolist()}")
d_out = (res["torch"][0] - res["cuda"][0]).abs().max().item()
d_grad = ((res["torch"][1] - res["cuda"][1]).norm() / res["torch"][1].norm()).item()
print(f"max|logit diff| = {d_out:.2e} | relative gradient diff = {d_grad:.2e}")
print("OK: CUDA kernel matches the PyTorch scan" if d_out < 1e-3 and d_grad < 1e-2 else "MISMATCH - do not use the CUDA backend")
