import sys, torch, warnings; warnings.filterwarnings("ignore")
import os as _os; sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from resnet_mamba_kan import KANLayer, ResNetMambaKAN
from resnet_mamba_mlp import ResNetMambaMLP
torch.manual_seed(0)
k = KANLayer(512, 6)
print("knots:", k.grid.numel(), "(expected G+2r+1 = 12);  bases per input:", k.spline_weights.shape[-1], "(expected G+r = 8)")
assert k.grid.numel() == 12 and k.spline_weights.shape[-1] == 8
x = torch.linspace(-2.99, 2.99, 1001).unsqueeze(1)                       # inside the grid range [-3, 3]
bs = k.b_splines(x)
print("partition of unity inside [-3,3]: min/max of sum_k B_k(x) =", bs.sum(-1).min().item(), bs.sum(-1).max().item())
assert torch.allclose(bs.sum(-1), torch.ones(1001, 1), atol=1e-5) and (bs >= -1e-6).all()
print("at most", (bs > 1e-9).sum(-1).max().item(), "non-zero bases per x (expected r+1 = 4)")
# initial logits should be of moderate scale for unit-variance features
feat = torch.randn(256, 512); out = k(feat)
print("initial logit std for N(0,1) features:", out.std().item())
assert out.std() < 1.0
# differentiable, finite
out.sum().backward(); assert all(torch.isfinite(p.grad).all() for p in k.parameters())
for cls in (ResNetMambaMLP, ResNetMambaKAN):
    m = cls(6, pretrained=False); y = m(torch.randn(2, 3, 224, 224)); print(cls.__name__, tuple(y.shape), "params", f"{sum(p.numel() for p in m.parameters()):,}", "head", m.describe()["head_parameters"])
    assert y.shape == (2, 6) and torch.isfinite(y).all()
# backbone is the SAME class/params in both hybrids
a, b = ResNetMambaMLP(6, pretrained=False), ResNetMambaKAN(6, pretrained=False)
na = {k_: v.shape for k_, v in a.backbone.state_dict().items()}; nb = {k_: v.shape for k_, v in b.backbone.state_dict().items()}
assert na == nb; print("MLP and KAN hybrids share an identical backbone (", len(na), "tensors )")
print("KAN / HYBRID CHECKS PASSED")
