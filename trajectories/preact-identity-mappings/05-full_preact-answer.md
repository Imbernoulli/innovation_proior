**Problem.** ReLU-only pre-activation fixed both previously-diagnosed hazards (clean highway,
unrestricted branch range) yet landed as a wash against baseline (6.71/5.91 vs. 6.61/5.93) —
ruling those two fixes out as sufficient alone and pointing at a third mechanism: normalization
placement. Tracing BN through every variant so far shows no weight layer in the network has
actually been receiving a BN-normalized input — BN's own normalization gets undone by the very
addition the architecture is built around (`x_l + BN(F(x_l))` is not itself normalized), and the
ReLU-only variant didn't change this since it left BN after each conv.

**Key idea.** Push BN forward too, in the order that actually delivers normalization to the
conv: `BN -> ReLU -> conv`, twice, nothing after the addition ("full pre-activation"). Verified
by direct construction (not just diagram): `max|y - (x + branch)| = 0.0` (the after-add op is
exactly identity) and the branch output spans negatives (`min=-0.24, max=+0.12` on a test
tensor) — both properties satisfied simultaneously, and now every weight layer's first operation
is BN, closing the normalization gap.

**Two distinct predicted payoffs.** (1) Optimization ease from the exact highway — should show
as faster/lower training loss, most visible early in training and at greater depth. (2)
Regularization from BN now properly conditioning every weight layer's input — should show as a
narrower train-test gap (test error down relative to training error), possibly *without* lower
training loss, since a more properly regularized model can converge to a higher training loss.
These point in different directions on the training-loss axis, which is how to tell them apart
if both effects are real.

**Step-1 edit.** Full pre-activation for both weight layers; shape-changing units share the
same leading `BN -> ReLU` between the projection and the branch (rather than a separate
activation for the projection); two boundary fixes forced by shifting every activation forward
by one unit — an activation right after the stem before the first split, and one extra
`BN -> ReLU` after the last addition, before pooling. Tested on ResNet-110 and ResNet-164.

**What to watch.** First variant in this line to clearly beat baseline on both architectures,
since every previously-derived necessary property is now satisfied at once with no
previously-diagnosed failure mode present.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    """Full pre-activation residual block: BN -> ReLU -> Conv, twice, with a
    clean (identity) after-addition path so same-shape identity stretches have
    the additive forward/backward highway x_L = x_l + sum F."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)

        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False)

    def forward(self, x):
        pre = F.relu(self.bn1(x))                 # first pre-activation
        out = self.conv1(pre)                      # BN -> ReLU -> Conv
        out = self.conv2(F.relu(self.bn2(out)))     # BN -> ReLU -> Conv (ends in conv: F unbounded)
        shortcut = self.shortcut(pre) if self.shortcut is not None else x  # shared pre-activation feeds the projection
        return shortcut + out                        # clean add, nothing after it


# Boundary fixes forced by shifting every interior activation forward by one
# unit: a stem activation before the first split, and a final BN->ReLU
# before pooling (no next unit exists to supply either on its own).
class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.stage1 = self._make_stage(block, 16, num_blocks[0], stride=1)
        self.stage2 = self._make_stage(block, 32, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(block, 64, num_blocks[2], stride=2)
        self.bn_final = nn.BatchNorm2d(64 * block.expansion)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_stage(self, block, planes, n, stride):
        strides = [stride] + [1] * (n - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)          # stem: first unit's shared BN->ReLU (inside block) activates this
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = F.relu(self.bn_final(out))   # extra BN->ReLU: no next unit exists to supply it
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)
```
