BN-after-addition landed exactly where the shortcut-sweep argument predicted — worse than
baseline on both ResNet-110 (8.17% vs. 6.61%) and ResNet-164 (6.50% vs. 5.93%), with elevated
early-training loss matching the sweep's failure signature. That's a real confirmation: the
after-addition operation `f` reaches the same shortcut-connected path the sweep already
condemned, because `f`'s output becomes the next unit's identity-shortcut input. `f` is not a
separate design question from the shortcut sweep — it's the same question, one step removed.

So the target is now unambiguous: make `f` identity, for the same reason `h` already is
identity, so the additive highway `x_L = x_l + sum F` and the backward direct term `dE/dx_l =
dE/dx_L * (1 + ...)` hold exactly rather than approximately. Right now `f` is ReLU. I can't
delete it outright — the network needs its nonlinearities — so the question is where to
relocate it, not whether to remove it. The most conservative move, changing exactly one thing,
is to relocate the single existing after-addition ReLU to immediately before the addition
instead of after it: `x_{l+1} = ReLU(F(x_l)) + x_l` instead of `x_{l+1} = ReLU(F(x_l) + x_l)`.
The addition is now bare, so `f` is exactly identity.

I want to flag a specific risk in this exact relocation before testing it, because it's a
structural property of *where* I put the ReLU, not a vague worry. Putting ReLU as the branch's
final operation means `F(x_l) = ReLU(...)`, which forces `F >= 0` as a hard constraint on the
function class — not a training-difficulty story but a capacity restriction. Two consequences
follow just from that constraint, without running anything: along a same-shape stretch, `x_L =
x_l + sum F` becomes a sum of exclusively non-negative terms, so the running feature can only
grow or stay flat with depth, never shrink; and the branch has lost its ability to *decrease* a
feature relative to the shortcut, which cuts directly against the premise that motivated fitting
`F(x) = H(x) - x` in the first place — that framing only helps if `F` can be small in either
direction around the identity target.

So this variant carries two effects pulling opposite ways, and I don't have a clean prior on
which wins. In its favor: the highway is now genuinely clean, and the compounding-truncation
hazard I diagnosed in the original unit's post-add ReLU should be gone. Against it: a real
capacity restriction, independent of how well SGD can optimize. The shortcut sweep taught me to
read training-error-vs-test-error as the signature of an optimization-difficulty failure; this
is a different kind of risk — closer to a converged model simply being unable to represent the
right function — and I'd expect it to look different in kind from the sweep's failures if it
does hurt. I'm testing on both ResNet-110 and ResNet-164 again, same recipe, single relocation,
nothing else in the branch or shortcut touched. This is the naive attempt at making `f` identity
by the shortest possible move, and it earns its place precisely because it isolates whether the
clean highway alone is sufficient, before I try anything that protects the branch's negative
range as well.

```python
import torch.nn as nn
import torch.nn.functional as F


class ReLUBeforeAddBlock(nn.Module):
    """ReLU relocated to the end of the branch (before the add) instead of
    after it. f is now exactly identity; F is constrained to [0, inf)."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        out = F.relu(out)                          # ReLU is now the branch's last op: F >= 0
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        return shortcut + out                       # bare add: f is exactly identity
```
