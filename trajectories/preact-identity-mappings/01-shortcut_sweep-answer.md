**Problem.** The current unit's shortcut is a bare identity, the cheapest possible choice. A
richer shortcut (gate, 1x1 conv) strictly contains identity in its solution space, so
naively it should only help. Before trusting that intuition, check what a non-identity `h`
costs the gradient highway that the additive shortcut is supposed to provide.

**Key idea / derivation.** Writing the unit as `x_{l+1} = h(x_l) + F(x_l, W_l)` and unrolling a
same-shape stretch, the backward pass carries a direct term `dE/dx_l = dE/dx_L * (prod h'_i + ...)`.
With `h` identity this product is exactly 1 at any depth — the highway. With `h(x) = lambda*x`,
the product becomes `prod lambda_i`, and at L=1000 even `lambda=0.99` decays it to `4.3e-5`
(`lambda=0.9` to `1.7e-46`) — any systematic non-unit shortcut factor compounds catastrophically
with depth, regardless of whether that shortcut's solution space contains identity as a special
case. That is an optimization argument (can SGD get there), not a representational one (can the
family express it) — and it needs testing against real learned gates and initializations, not
just a fixed scalar, since a gate could in principle learn to sit near the safe corner.

**Step-1 edit.** Sweep the shortcut function `h` on ResNet-110 / CIFAR-10, `F` (two 3x3
conv-BN, ReLU after addition) and everything else frozen: constant scaling (`lambda=0.5`, with
and without also scaling `F` by `1-lambda`); exclusive gating (`h`=`1-g(x)`, `F` scaled by
`g(x)`, `g(x) = sigmoid(W_g x + b_g)`, `b_g` swept from 0 to -7); shortcut-only gating (`h` =
`1-g(x)`, `F` unscaled, `b_g` swept); a 1x1 convolutional shortcut on every unit; dropout
(p=0.5) on the shortcut output.

**Why test rather than assert.** The lambda-product algebra gives a strong prior that every
richer shortcut hurts, but it is derived for `h` alone on a same-shape stretch, isolates only
one of the two open design variables (the after-addition ReLU is untouched here), and a
*learned* gate is a different object from a fixed unfavorable scalar. If the algebra
overstates the risk, a well-initialized gate or the 1x1 conv should be able to match or beat
6.61%.

**What to watch.** If the highway argument is right: every variant at or worse than baseline,
severity tracking how far the variant's effective multiplicative factor sits from 1, gated
variants showing strong sensitivity to gate-bias initialization (closer to `1-g(x)=1` at
init should track closer to baseline). If wrong: at least one variant should clearly beat
6.61%, favoring the representational-capacity story over the optimization-path story.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortcutSweepBlock(nn.Module):
    """Residual block with a swept shortcut function h(x); F and the
    after-addition ReLU are the unchanged original-unit choices."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut_type="identity",
                 scale_lambda=0.5, gate_bias=-6.0, dropout_p=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut_type = shortcut_type
        self.scale_lambda = scale_lambda
        self.dropout_p = dropout_p
        self.needs_proj = (stride != 1 or in_planes != planes * self.expansion)

        if shortcut_type == "conv1x1" or self.needs_proj:
            self.proj = nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False)
        if shortcut_type in ("exclusive_gate", "shortcut_only_gate"):
            self.gate = nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=True)
            nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, x):
        f_out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))  # unchanged residual branch

        if self.shortcut_type == "identity":
            h = self.proj(x) if self.needs_proj else x
            return F.relu(h + f_out)

        if self.shortcut_type == "const_scale":
            h = self.scale_lambda * (self.proj(x) if self.needs_proj else x)
            return F.relu(h + f_out)                       # F unscaled

        if self.shortcut_type == "const_scale_both":
            base = self.proj(x) if self.needs_proj else x
            h = self.scale_lambda * base
            return F.relu(h + (1 - self.scale_lambda) * f_out)

        if self.shortcut_type in ("exclusive_gate", "shortcut_only_gate"):
            base = self.proj(x) if self.needs_proj else x
            g = torch.sigmoid(self.gate(x))
            h = (1 - g) * base
            branch = g * f_out if self.shortcut_type == "exclusive_gate" else f_out
            return F.relu(h + branch)

        if self.shortcut_type == "conv1x1":
            h = self.proj(x)                                # 1x1 conv on every unit
            return F.relu(h + f_out)

        if self.shortcut_type == "dropout":
            base = self.proj(x) if self.needs_proj else x
            h = F.dropout(base, p=self.dropout_p, training=self.training)
            return F.relu(h + f_out)

        raise ValueError(self.shortcut_type)
```
