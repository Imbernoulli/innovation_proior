Full pre-activation is the first design in this sequence to clearly beat the original unit,
holding across every architecture tested so far — 6.37% vs. 6.61% on ResNet-110, 5.46% vs. 5.93%
on ResNet-164, and the same direction on a third, one-conv-per-unit configuration (8.91% vs.
9.90%). Every property I derived as necessary is satisfied at once, and none of the three
failure modes diagnosed along the way is present. But none of that has touched the regime this
whole investigation actually started from.

The question that opened this work was about extreme depth specifically: a 1202-layer residual
net using the original unit fits its training set extremely well and still tests worse than a
110-layer net using the same unit, and separately, the deepest residual nets built so far show
training loss falling only slowly at the start. Neither observation is about 110 or 164 layers.
Everything tested since — the shortcut sweep, the activation-placement experiments, full
pre-activation itself — has been at a depth where the original unit's after-add ReLU gating
hazard should be comparatively mild: each unit's input in the original design is itself the
output of a ReLU, hence already non-negative, so the pre-add sum is only occasionally negative
enough to trip the gate. At 100-ish layers, that's a survivable amount of occasional truncation.
At 1000+ layers, the same occasional-per-unit truncation compounds across an order of magnitude
more units — exactly the regime where a small per-unit effect should become the dominant one. So
if the identity-highway mechanism is really the right explanation for why very deep residual
nets are hard to train well, I'd expect the fix to matter *more*, not less, at the depth the
original anomaly actually lives at than at the depths tested so far.

That's the natural next test: push full pre-activation to 1000+ layers and see whether the
improvement grows the way the mechanism predicts, or stays at the modest margin measured at
110/164. Before I can run it, there's a practical problem independent of pre-activation itself:
the two-3x3-conv branch used everywhere so far, stacked to a thousand-plus layers at full width,
burns through a parameter count a 50,000-image training set can't usefully support. I already
have a cheaper branch shape for this: the bottleneck unit, where a 1x1 conv first reduces the
channel count, the expensive 3x3 runs at that reduced width, and a second 1x1 restores a wider
output — four times the reduced width. Because the costly kernel never touches full width, this
reaches a substantially wider output at roughly the parameter budget of a narrower two-3x3
branch, keeping the parameter growth from swamping a fixed, modest-sized dataset.

Switching branch shape shouldn't change the pre-activation argument itself — the whole
derivation, from the shortcut sweep through to full pre-activation, was a statement about
branch-and-merge *topology* (an activation in front of every weight layer in the branch, the
merge left bare), never about how many weight layers sit inside the branch. So the substitution
is mechanical: the same `BN -> ReLU` in front of each of the bottleneck's three weight layers
(the reducing 1x1, the 3x3, the restoring 1x1), the downsampling stride placed on the first 1x1
so the 3x3 runs on the smaller feature map, everything else — bare add, shared pre-activation
feeding the projection shortcut, stem and final-layer boundary activations — exactly as derived
for the basic block.

Concrete proposal: build a 1001-layer network with this pre-activated bottleneck unit and
compare it against a 1001-layer network built the same way with the original unit, same
training recipe, same protocol as every comparison in this sequence, on both CIFAR-10 and
CIFAR-100 — CIFAR-100 as an independent second read on whether generalization improves rather
than just training fit. My prediction: if the identity-highway explanation is right, the
original-unit network at this depth should reproduce the slow-start training symptom and
plausibly something like the 1202-layer generalization anomaly this investigation opened with,
while the pre-activated version should show training loss falling quickly from the start, since
none of the accumulated after-add-ReLU truncation is present and the direct gradient term is now
exact rather than gated at each of a thousand-plus units. If the roughly half-point margin from
110/164 layers simply carries forward unchanged, that would say the depth-dependent compounding
story is weaker than the derivation suggests; if it widens substantially instead — my actual
expectation, given how directly the mechanism should scale with the number of units it
compounds across — that would be the clearest confirmation that the identity-highway argument is
what drives the improvement, and it would resolve the anomaly this whole line of work started
from: a very deep residual network that both trains easily and generalizes at least as well as a
much shallower one.

```python
import torch.nn as nn
import torch.nn.functional as F


class PreActBottleneck(nn.Module):
    """Pre-activation bottleneck: 1x1 (reduce) -> 3x3 -> 1x1 (restore), each
    preceded by BN -> ReLU. Same clean-add / shared-preactivation-shortcut
    logic as PreActBlock, spread over three weight layers instead of two."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)

        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False)

    def forward(self, x):
        pre = F.relu(self.bn1(x))
        shortcut = self.shortcut(pre) if self.shortcut is not None else x
        out = self.conv1(pre)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        return shortcut + out


def make_resnet1001(num_classes=10):
    """333 pre-activated bottleneck units, 111 per feature-map size
    (16/32/64 base widths, expansion=4), plus stem conv and the final
    BN->ReLU before pooling (same boundary fixes as PreActResNet)."""
    return PreActResNet(PreActBottleneck, num_blocks=[111, 111, 111], num_classes=num_classes)
```

(`PreActResNet` is the backbone from the previous rung, unchanged — stem conv, three stages, a
final `BN -> ReLU` before global average pooling.)
