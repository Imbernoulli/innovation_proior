The residual shortcut in the unit I'm working with is a bare identity — no parameters, no
computation, just the input added back to the transformed branch. That is the cheapest possible
choice, and cheapest is not obviously best: a gated shortcut or a 1x1-convolution shortcut is
strictly more expressive, since either can represent identity as one point in its solution space
and everything else besides. The field already trusts richness on this connection in two places
— highway networks put a learned gate directly on it, and the shortcut is already allowed to be
a 1x1 projection wherever channel counts change — so the natural next question is whether
richness helps everywhere, not just where shape forces a projection.

Before running anything I wanted to know what the algebra says, because if there is a reason
richness should hurt, I want the shape of that reason before spending a training run to
rediscover it. Write the unit in general form as `x_{l+1} = h(x_l) + F(x_l, W_l)`. With `h` the
identity, unrolling a same-shape stretch of the network gives `x_L = x_l + sum_i F(x_i, W_i)` —
an additive relationship with `x_l` sitting out front undistorted, unlike a plain net's product
of weight matrices, which is exactly the kind of thing that explodes or vanishes with depth. The
backward pass mirrors this: `dE/dx_l = dE/dx_L * (1 + [residual term])`, a direct gradient
highway with a leading coefficient of exactly 1 that cannot vanish regardless of what the weight
layers do.

Now perturb `h`. With a constant scale `h(x) = lambda * x`, that leading coefficient becomes a
product of `L-l` scale factors, `prod lambda_i`. At depth 1000, `lambda=0.9` collapses the
product to `1.7e-46` and `lambda=1.1` blows it to `2.5e+41`; even `lambda=0.99`, barely off
unity, decays it to `4.3e-5` — enough to switch the highway off in practice. Only `lambda=1` is
stable. The same logic generalizes to any smooth non-identity `h`: the scalar product becomes a
product of shortcut Jacobians, and any systematic departure from identity compounds the same way
over enough layers. Crucially, this is an argument about the *optimization path*, not about
representational capacity — it says nothing about whether a gate or a 1x1 kernel can express
something better than identity, only that getting there through SGD, with the gradient signal
intact, is harder the further the shortcut's factor sits from 1 during training.

That argument is a strong prior, not a proof, for two reasons: it isolates `h` alone, on a
same-shape stretch, leaving the after-addition ReLU (the network's other non-identity operation
near the shortcut) untouched; and it is derived for a fixed scalar, while a *learned* gate could
in principle settle near the safe corner on its own, especially with the initialization the
highway-network literature already recommends biasing toward. So I am proposing a direct sweep
rather than trusting the recursion alone: constant scaling on the shortcut (with and without
correspondingly scaling `F`); an exclusive gate in the highway style, `h = (1-g(x))*x`,
`F` scaled by `g(x)`, with the gate bias `b_g` swept since initialization is flagged as critical
in the gating literature; a shortcut-only gate, `h = (1-g(x))*x` with `F` left unscaled, to
isolate the shortcut-side effect from the coupling the exclusive form introduces; a 1x1
convolutional shortcut applied to every unit, generalizing the already-trusted shape-change
projection to the whole network; and dropout on the shortcut output, a different flavor of
stochastic attenuation with the same expected scale as the failed constant-scaling case.
Everything else — ResNet-110 on CIFAR-10, the training recipe, `f` = ReLU after the addition,
median of 5 runs — stays exactly as currently used, so the shortcut function is the only
variable in play.

If the lambda-product story is right, I expect every variant here to land at or worse than the
6.61% baseline, with the damage tracking how far each variant's effective multiplicative factor
sits from exactly 1, and the gated variants showing clear sensitivity to how the gate bias is
initialized. If the richer-shortcut intuition is right instead, at least one of these — most
plausibly the 1x1 convolution, or a well-initialized gate — should be able to match or beat the
baseline, since each strictly contains identity in its solution space. Either outcome tells me
something I need before touching the other open variable in this unit, the ReLU that currently
sits after the addition.

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
