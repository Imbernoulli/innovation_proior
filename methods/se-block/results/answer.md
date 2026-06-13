# Squeeze-and-Excitation block, distilled

The Squeeze-and-Excitation (SE) block is a lightweight channel-attention unit that lets a
convolutional network use **global** context to **adaptively, per input** reweight its feature
channels. Given feature maps `U ∈ R^{H×W×C}`, it (1) **squeezes** each channel's spatial map to
one scalar by global average pooling, (2) **excites** a set of per-channel gates from that
descriptor with a small bottleneck MLP ending in a sigmoid, and (3) **rescales** each channel by
its gate. It is a drop-in unit: stacked into a backbone (SE-ResNet, SE-ResNeXt, SE-Inception)
it improves accuracy at a slight parameter and compute cost.

## Problem it solves

An ordinary convolution mixes channels only implicitly (the weighting is fused into the
filters), locally (each response sees a small receptive field), and statically (the same
combination is applied to every input after training). There is no mechanism for a unit to use
global information to emphasise informative channels and suppress less useful ones on a
per-input basis. The SE block adds exactly that mechanism without disturbing the rest of the
backbone.

## Key idea

Recalibrate channels with an input-conditioned multiplicative gate:

- **Squeeze (global information embedding).** Aggregate each channel over space:

  ```
  z_c = F_sq(u_c) = (1 / (H·W)) · sum_{i=1}^{H} sum_{j=1}^{W} u_c(i, j),    z ∈ R^C
  ```

  Global average pooling gives one global descriptor per channel — the global context a local
  convolution cannot access. (Average over max: a smooth full-map summary rather than a single
  brittle peak.)

- **Excitation (adaptive recalibration).** Map the descriptor to per-channel gates with a
  two-layer bottleneck MLP and a sigmoid:

  ```
  s = F_ex(z, W) = σ( W_2 · δ( W_1 z ) ),   W_1 ∈ R^{(C/r)×C},  W_2 ∈ R^{C×(C/r)},  δ = ReLU
  ```

  - **Sigmoid, not softmax**, because the gates must be *non-mutually-exclusive*: many channels
    can be informative at once, so each `s_c ∈ (0,1)` is set independently rather than competing
    on a simplex.
  - **Bottleneck (reduce by `r`, ReLU, expand back)**, because a full `C×C` map is too many
    parameters per block and overfits; the squeeze to `C/r` cuts params to `2C^2/r` per block and
    regularises the gate toward a low-rank model of channel interdependence.

- **Scale (recalibration).** Multiply each channel by its gate:

  ```
  x~_c = F_scale(u_c, s_c) = s_c · u_c.
  ```

  Multiplicative (not additive) so it attenuates or preserves existing features instead of
  injecting a separate additive signal; near-one gates recover the unscaled features.

## Design choices and why

| Choice | Why this, not the alternative |
|---|---|
| Recalibrate channels | The conv already models spatial structure; the channel axis is the under-modelled one (implicit, local, static). |
| Multiplicative gate `s_c·u_c` | Additive injects a new feature instead of recalibrating the existing one; sigmoid-bounded multiplicative gates attenuate or pass channels, with an all-ones no-op limit. |
| Squeeze = global average pool | Need one global scalar per channel; the mean is a smooth full-image summary, max keeps one noisy peak (GAP slightly better, both work). |
| Excitation = nonlinear MLP | Channel importance is a nonlinear function of the global descriptor capturing interdependence, not a linear per-channel readout. |
| Sigmoid (not softmax) | Non-mutually-exclusive gating: multiple channels emphasisable independently; softmax forces competition (one up → others down). |
| Bottleneck FC, ratio `r` | Full `C×C` is too costly and overfits; reduce→ReLU→expand cuts params to `2C^2/r` and regularises. |
| `r = 16` | Sits in the flat region of the accuracy/cost trade-off; robust across `r∈{2..32}`, smaller `r` not monotonically better and balloons params (mostly in the top stage). |
| SE on the residual branch, **before** the identity add | Keep the always-open identity/gradient path clean; gating after the summation would attenuate the skip and hurt. |

Total extra parameters: `(2/r) · Σ_s N_s · C_s^2` (stages `s`, `N_s` blocks, `C_s` channels) —
concentrated in the final stage; compute overhead is a global pool plus two small matrix-vector
products plus a channel-wise scale (negligible).

## Integration into a residual block

The SE block recalibrates the residual (non-identity) branch; squeeze and excitation both act
before the summation with the identity:

```
x → conv → BN → ReLU → conv → BN → [ SE: GAP → FC↓ → ReLU → FC↑ → sigmoid → scale ] → + identity → ReLU
```

## Working code

A CIFAR basic residual block (`expansion = 1`) with SE recalibration, matching the fixed
backbone interface `(in_planes, planes, stride)`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        z = self.avg_pool(x).view(b, c)
        s = self.fc(z).view(b, c, 1, 1)
        return x * s.expand_as(x)


class SEResidualBlock(nn.Module):
    """Basic residual block with Squeeze-and-Excitation channel attention."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

        self.se = SELayer(planes, reduction=16)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))              # residual-branch features U
        out = self.se(out)                           # GAP -> FC↓ -> ReLU -> FC↑ -> sigmoid -> scale
        out += self.shortcut(x)                      # untouched identity path
        return F.relu(out)
```

The `SELayer` helper is the standalone SE module: it can be inserted after the last BN of a
basic or bottleneck branch and before the residual addition.
