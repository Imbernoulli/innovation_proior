**Problem.** Full pre-activation clearly beat baseline at 110/164 layers, but every experiment
so far has been at a depth where the original unit's after-add ReLU gating hazard is
comparatively mild (each unit's input is already non-negative from the previous unit's ReLU, so
the pre-add sum is only occasionally negative). The regime this whole investigation actually
started from — a 1202-layer net generalizing worse than a 110-layer net despite fitting training
data well, and 1000-layer training loss falling only slowly at the start — is 1000+ layers,
untouched so far.

**Key idea.** Push the same full pre-activation substitution to that depth. The derivation was
always about branch-and-merge *topology* (activation in front of every weight layer, bare
merge), never about how many weight layers sit inside the branch — so the substitution carries
over mechanically to a bottleneck branch (1x1 reduce -> 3x3 -> 1x1 restore), which is the right
shape at this depth for a practical reason independent of pre-activation: a full-width two-3x3
branch at 1000+ layers burns through a parameter budget a 50k-image dataset can't usefully
support, while a bottleneck branch reaches a wider output at roughly the same parameter cost,
since the costly 3x3 never touches the full width.

**Step-1 edit.** 1001-layer network, pre-activated bottleneck unit throughout (same `BN -> ReLU`
in front of each of the three weight layers, downsampling stride on the first 1x1, same bare
add, same shared pre-activation feeding the projection shortcut, same stem/final boundary
activations as derived for the basic block). Compared against a 1001-layer network built the
same way with the *original* unit. CIFAR-10 and CIFAR-100, same protocol throughout.

**Prediction.** If the identity-highway mechanism is really the explanation for the
depth-dependent training difficulty, the margin over the original unit should *widen*
substantially relative to the roughly half-point margin measured at 110/164 layers — the
after-add ReLU truncation the original unit accumulates has a thousand-plus units to compound
across here instead of a hundred-odd. If the margin instead stays flat, the depth-dependent
compounding story is weaker than the derivation implies.

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
