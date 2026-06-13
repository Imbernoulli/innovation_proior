# Pre-activation residual unit, distilled

The pre-activation residual unit reorders a residual block to **BN → ReLU → Conv** (instead of
Conv → BN → ReLU), so that the element-wise addition is followed by *no* activation. This makes
the after-addition mapping an exact identity, which — together with the parameter-free identity
shortcut — turns the network into an exact additive highway between any two layers, in both the
forward and backward pass. It is a zero-extra-parameter change to the building block that makes
very deep residual nets (hundreds to ~1000 layers) easier to optimize and better-generalizing.

## Problem it solves

Residual learning eased the *degradation* problem (deeper plain nets have higher *training*
error), but the original residual unit `x_{l+1} = ReLU(F(x_l) + x_l)` applies ReLU *after* the
addition. That after-add ReLU sits on the shortcut path and intermittently severs it; the
effect is mild at moderate depth but compounds at extreme depth, where the very deep net trains
slowly and can underperform a shallower one. The goal: make the path between any two layers as
undistorted as possible, with no extra parameters and no change to the training recipe.

## Key idea

Write a unit as `y_l = h(x_l) + F(x_l, W_l)`, `x_{l+1} = f(y_l)`. If **both** `h` and `f` are
identity, then

```
forward:   x_L = x_l + sum_{i=l}^{L-1} F(x_i, W_i)        # a SUM, not a product of weights
backward:  dE/dx_l = dE/dx_L * ( 1 + d/dx_l sum F )       # the "1" is a never-vanishing path
```

- **Identity shortcut `h` is necessary.** Replacing `h(x_l)=x_l` with `h(x_l)=lambda_l x_l`
  makes the forward leading term `(prod lambda_i) x_l` and the backward direct term
  `(prod lambda_i) dE/dx_L`. Over many layers this product explodes (`lambda_i>1`) or vanishes
  (`lambda_i<1`). The same multiplicative trap (`prod h'_i`) condemns gating, 1x1-conv, and
  dropout shortcuts: they add representation but break optimization, so identity is kept.
- **Identity after-add `f` is the missing piece.** With `f = ReLU`, the backward "1" is gated
  by `ReLU'(y_l)` (zero where `y_l < 0`), so the highway is approximate and frays at extreme
  depth. To make `f` identity *without deleting any activation*, make the after-add activation
  asymmetric so it touches only the residual branch: `x_{l+1} = x_l + F(f_hat(x_l), W_l)`. This
  asymmetric activation **is** the pre-activation of the next unit's weight layers.
- **Full pre-activation (BN-ReLU-Conv) is the right ordering**, isolated by elimination:
  - *BN after addition* puts BN on the clean shortcut → breaks the path.
  - *ReLU before addition* makes `F >= 0`, so the forward signal is monotonically
    non-decreasing along depth → a residual should range over `(-inf, +inf)`; rejected.
  - *ReLU-only pre-activation* leaves the leading ReLU un-paired with a BN → no BN benefit.
  - *Full pre-activation* `BN -> ReLU -> Conv`: after-add is identity (exact highway, eased
    optimization, benefit growing with depth) **and** the first op of every weight layer is a
    BN, so every weight layer gets a normalized input — delivering BN's regularization that the
    original post-add unit was undoing (it normalized `F` then added the un-normalized shortcut,
    leaving the next weight layer's input un-normalized).

Two effects, two mechanisms: **ease of optimization** comes from `f` = identity (faster, lower
training loss, especially deep); **regularization** comes from BN-at-the-front (slightly higher
training loss but lower test error).

## Final block

Basic residual branch = two 3x3 convs in pre-activation order, a clean add with nothing after
it, an identity shortcut (or a 1x1 projection fed from the shared pre-activation, used **only**
for dimension changes), and **no** residual scaling. Two boundary units are forced by the
asymmetric-shift construction: a `BN -> ReLU` right after the stem conv (before the first unit
splits), and an extra `BN -> ReLU` after the last unit's addition (before global average pooling
and the linear classifier).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    """Full pre-activation basic residual block (BN -> ReLU -> Conv, twice)."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)

        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            # 1x1 projection ONLY for dimension changes; fed from the shared pre-activation.
            self.shortcut = nn.Conv2d(
                in_planes, planes * self.expansion, 1, stride=stride, bias=False)

    def forward(self, x):
        pre = F.relu(self.bn1(x))                  # shared first pre-activation
        out = self.conv1(pre)
        out = self.conv2(F.relu(self.bn2(out)))    # branch ends in a conv => F can be negative
        shortcut = self.shortcut(pre) if self.shortcut is not None else x
        return shortcut + out                      # clean add; no activation after it
```

Pre-activation bottleneck unit (for the deeper CIFAR/ImageNet ResNets) is the same idea with
1x1 → 3x3 → 1x1 convs, each in `BN -> ReLU -> Conv` order:

```python
class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)

        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(
                in_planes, planes * self.expansion, 1, stride=stride, bias=False)

    def forward(self, x):
        pre = F.relu(self.bn1(x))
        shortcut = self.shortcut(pre) if self.shortcut is not None else x
        out = self.conv1(pre)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        return shortcut + out
```

Backbone boundary handling: `BN -> ReLU` after the stem conv (before the first unit) and a
final `BN -> ReLU` after the last unit, before global average pooling and the classifier.

## Why it works (one line each)

- Identity shortcut + identity after-add ⇒ exact additive forward path and an exact `+1`
  backward path ⇒ signal/gradient propagate undistorted between any two layers.
- ReLU after the add gates that `+1` where the pre-add sum is negative; pre-activation removes
  it, and the benefit grows with depth (truncation is rare at ~100 layers, frequent at ~1000).
- BN at the front of every weight layer (not undone by the addition) delivers BN's
  regularization to every weight layer's input.
