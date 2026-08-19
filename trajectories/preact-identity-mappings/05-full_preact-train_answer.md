ReLU-only pre-activation landed as a wash against baseline — 6.71% vs. 6.61% on ResNet-110,
5.91% vs. 5.93% on ResNet-164, close enough to call a tie. That's informative rather than
disappointing: both previously-diagnosed hazards (the shortcut-compounding from a merge-point
BN, the capacity restriction from a branch-ending ReLU) are absent in this variant, and yet
nothing improved. That rules out "clean highway plus unrestricted branch range" as sufficient
on its own and leaves a third mechanism as the remaining candidate — where BN actually sits.

Trace BN through every variant so far rather than just noting its presence. In the original
unit, the branch ends `conv-BN`, so the second BN normalizes the branch's own output — but that
normalized output is immediately summed with the raw, un-normalized shortcut `x_l`. The sum
`x_l + BN(F(x_l))` is not itself normalized, and that unnormalized sum is exactly what feeds the
next unit's first weight layer. Despite BN sitting right there in every unit, no weight layer in
the network actually receives a BN-normalized input. The ReLU-only variant didn't touch this:
its leading ReLU operates on raw `x_l` (since `f` is identity, `x_l` *is* the previous unit's
un-normalized sum), and BN is still positioned after each conv, still normalizing intermediate
branch outputs rather than any weight layer's actual input.

The fix is direct: if every weight layer should receive a normalized input, BN has to be the
*first* operation applied to what feeds that layer — in front of the conv, and in front of the
ReLU. Push BN forward the same way I pushed ReLU forward, in the order that actually delivers
normalization to the conv: BN, then ReLU (also BN's originally-intended use, immediately before
a nonlinearity), then conv. Each branch becomes `BN -> ReLU -> conv -> BN -> ReLU -> conv`,
nothing after the addition — full pre-activation.

I checked this delivers what I intend by direct construction rather than trusting the diagram:
building the block and comparing its output against the shortcut plus the branch computed
separately gives `max|y - (x + branch)| = 0.0` — the after-add operation really is a bare add.
And the branch's own output on the same input spans negatives (`min=-0.24, max=+0.12`),
confirming it isn't range-restricted. Both properties I've been chasing across three experiments
hold simultaneously, and now every weight layer's first operation is BN.

That gives two candidate benefits, worth keeping separate since they'd show different
signatures. Optimization ease, from the exact highway: forward `x_L = x_l + sum F` and backward
`dE/dx_l = dE/dx_L * (1 + ...)` hold exactly now, with nothing left to gate or scale the direct
term — I'd expect this to show as faster, lower training loss, most visible early in training
and at greater depth. Regularization, from BN now properly conditioning every weight layer's
input instead of only intermediate branch outputs: BN's regularizing effect comes from
mini-batch noise injected into each activation, and if that's now delivered where it should be,
I'd expect a narrower train-test gap — possibly *without* lower training loss, since a properly
regularized model can converge to a higher training loss. If training loss ends up lower, that
points at the highway story; if training loss is flat or slightly higher while test error still
improves, that points at regularization; both signatures together would mean both mechanisms
contribute.

Two boundary details, forced by shifting every interior activation forward by one unit rather
than chosen freely: dimension-changing units share the same leading `BN -> ReLU` between the
1x1 projection and the branch, rather than a separate activation for the projection; and since
the first unit has no predecessor (it follows a stand-alone stem conv) and the last unit has no
successor to supply a closing activation, I add one extra `BN -> ReLU` right after the stem and
one right before pooling. Testing on ResNet-110 and ResNet-164 again, matching every previous
experiment. My prediction is that this should be the first variant in this line to clearly beat
baseline on both architectures — it's the first to satisfy every property derived as necessary
(clean highway, unrestricted branch range, normalized weight-layer input) with none of the three
previously-diagnosed failure modes present. What I don't yet know is the relative size of the
two effects, or whether the depth-dependent optimization benefit is even visible at 110/164
layers rather than only becoming prominent at the extreme depths the original units are known to
struggle at.

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
