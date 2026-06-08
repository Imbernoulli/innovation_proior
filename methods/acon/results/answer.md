# ACON

**Problem.** Swish `x·σ(βx)` works but is an unexplained search result; its first-derivative bounds are
*fixed* and its gains shrink on deep/optimized nets. Find the principle behind Swish, generalize it to a
family that strictly contains Swish, and let the network learn — per channel and per sample, at
negligible cost — whether each neuron behaves nonlinearly (activate) or linearly (not).

**Key idea.** ReLU/LReLU/PReLU/abs are all `max(η_a(x), η_b(x))` of two linear pieces (Maxout). Apply
the standard **smooth maximum** `S_β(x_1,…,x_n)=Σ_i x_i e^{βx_i}/Σ_i e^{βx_i}`; the two-piece case
simplifies to

`S_β(η_a, η_b) = (η_a − η_b)·σ(β(η_a − η_b)) + η_b.`

With `η_a=x, η_b=0` this is `x·σ(βx)` = **Swish** — so Swish *is the smooth max of `{x,0}`* (smooth
ReLU), and `β` is a temperature: `β→∞` ⇒ `max` (nonlinear, "activate"), `β→0` ⇒ arithmetic mean
(linear, "not"). Smoothing the general member `max(p_1 x, p_2 x)` gives

`ACON-C:  f(x) = (p_1 − p_2)·x·σ(β(p_1 − p_2)x) + p_2 x`  (per-channel learnable `p_1, p_2, β`; the
point `p_1=1, p_2=0` recovers Swish).

**meta-ACON:** generate the switch from the input, `β = G(x)`, instead of storing only a fixed scalar.
Channel-wise SE-style routing `β = σ(BN_2(W_2(BN_1(W_1(GAP(x))))))` (reduction `r=16`) gives
per-sample, per-channel non-linear degree at negligible parameter cost; the implementation uses a
two-layer 1×1-convolution bottleneck with BatchNorm after each convolution before the sigmoid.

**Why it works.** For `p_1>p_2` and `β>0`, `ACON-C`'s derivative asymptotes to slopes `p_1` (`x→+∞`)
and `p_2` (`x→−∞`); its max/min (from `(y−2)e^y=y+2`, where `y=(p_1−p_2)βx` and the nonzero roots are
`y≈±2.39936`) are `1.0998 p_1 − 0.0998 p_2` and `1.0998 p_2 − 0.0998 p_1` — **learnable**
gradient upper/lower bounds, where Swish's (≈1.0998, ≈−0.0998) are *fixed* and `β` only sets how fast
the derivative reaches them. So `β` controls the non-linear degree and `p_1, p_2` control the gradient
bounds — two knobs Swish conflated. meta-ACON adds per-sample adaptivity, which is the specific
mechanism to test where a fixed curve has little remaining headroom. Channel-wise meta-ACON's routing
is the SE module; the activation additionally reshapes its gradient via `p_1, p_2`.

**Hyperparameters.** ACON-C: per-channel `p_1, p_2, β`, initialized as `p_1=1`, `p_2=0`, `β=1`
(an intentional SiLU-start choice; the official `acon.py` samples `p_1,p_2` from `randn`). meta-ACON:
SE bottleneck reduction `r=16`
with an implementation floor of `r` bottleneck channels; use ACON-C as the underlying form. Drop in
for ReLU/Swish.

```python
import torch
import torch.nn as nn


class AconC(nn.Module):
    """ACON-C: (p1-p2)*x*sigmoid(beta*(p1-p2)*x) + p2*x, per-channel p1,p2,beta."""

    def __init__(self, width):
        super().__init__()
        self.p1 = nn.Parameter(torch.ones(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, width, 1, 1))

    def forward(self, x):
        diff = (self.p1 - self.p2) * x
        return diff * torch.sigmoid(self.beta * diff) + self.p2 * x


class MetaAconC(nn.Module):
    """meta-ACON-C: beta generated from GAP(x) by a BN bottleneck."""

    def __init__(self, width, r=16):
        super().__init__()
        inner = max(r, width // r)
        self.fc1 = nn.Conv2d(width, inner, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(inner)
        self.fc2 = nn.Conv2d(inner, width, kernel_size=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm2d(width)
        self.p1 = nn.Parameter(torch.ones(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.zeros(1, width, 1, 1))

    def forward(self, x):
        ctx = x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        beta = torch.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(ctx)))))
        diff = (self.p1 - self.p2) * x
        return diff * torch.sigmoid(beta * diff) + self.p2 * x
```
