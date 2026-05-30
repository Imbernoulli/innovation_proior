# EfficientNet: compound model scaling

## Problem

Convolutional networks are scaled up for higher accuracy whenever more compute is available, but
the *how* is folklore: practitioners enlarge a single dimension — depth (more layers), width (more
channels), or resolution (bigger input) — or balance two or three by hand. Single-dimension scaling
saturates (each curve plateaus around the low-80s ImageNet top-1), and manual multi-dimension tuning
is tedious and yields sub-optimal accuracy/FLOPS trade-offs. Goal: a principled, reusable rule for
distributing a compute budget across all three dimensions, cheap to derive at any target size.

## Key idea

The three scaling dimensions are coupled — a higher-resolution image has more pixels per object, so
it needs a larger receptive field (more depth) and more channels (more width) to exploit them.
**Compound scaling** therefore moves all three together with one user knob φ and three constants:

  depth   d = α^φ
  width   w = β^φ
  resolution r = γ^φ      with   α ≥ 1, β ≥ 1, γ ≥ 1.

**The constraint α·β²·γ² ≈ 2.** The FLOPS of a regular convolution scale linearly with depth and
quadratically with width and with resolution (a conv layer costs k²·C_in·C_out·H·W: depth adds
layers ∝ d; width scales both C_in and C_out ∝ w²; resolution scales H·W ∝ r²). Total FLOPS thus
scale as d·w²·r² = (α^φ)(β^φ)²(γ^φ)² = (α·β²·γ²)^φ. Pinning α·β²·γ² ≈ 2 makes total FLOPS scale as
2^φ, so φ is a clean "double the compute φ times" dial. The quadratic weighting of β, γ vs. linear
α also explains why depth gets the largest exponent — it is the cheapest dimension per unit scaling.

## Two-step recipe

1. **Find the ratios once, on the small baseline.** Fix φ = 1 and grid-search α, β, γ subject to
   α·β²·γ² ≈ 2 on the small baseline (B0). Result: **α = 1.2, β = 1.1, γ = 1.15** (check:
   1.2·1.1²·1.15² = 1.92 ≈ 2). Searching ratios directly on large models is prohibitively
   expensive; doing it once on the small model and reusing the ratios amortizes the cost.
2. **Scale by sweeping φ.** Freeze α, β, γ and increase φ to produce the family B0→B7. Per-model
   (width, depth, resolution) coefficients: B0 (1.0, 1.0, 224), B1 (1.0, 1.1, 240), B2 (1.1, 1.2,
   260), B3 (1.2, 1.4, 300), B4 (1.4, 1.8, 380), B5 (1.6, 2.2, 456), B6 (1.8, 2.6, 528),
   B7 (2.0, 3.1, 600). Dropout rises 0.2→0.5 with size (larger models need more regularization).

## Baseline architecture (EfficientNet-B0)

Because scaling never changes the per-layer operator, the baseline sets the accuracy ceiling. B0 is
found by a FLOPS-aware multi-objective architecture search (maximize ACC(m)·(FLOPS(m)/T)^w, w ≈
−0.07, target T ≈ 400M FLOPS). Its block is the **mobile inverted bottleneck (MBConv)**: 1×1 expand
(×6) → k×k depthwise → squeeze-and-excitation gate → 1×1 project (linear), with a residual skip on
the thin endpoints when shapes match. The projection is left linear because a ReLU on the narrow
bottleneck would destroy information; SE recalibrates channels by global context at near-zero cost.
Activation is SiLU/Swish (x·σ(x)). Stages (expand, kernel, stride, in, out, repeats):

| Stage | op | resolution | channels | layers |
|---|---|---|---|---|
| stem | Conv3×3 s2 | 224² | 32 | 1 |
| 1 | MBConv1 k3 | 112² | 16 | 1 |
| 2 | MBConv6 k3 | 112² | 24 | 2 |
| 3 | MBConv6 k5 | 56² | 40 | 2 |
| 4 | MBConv6 k3 | 28² | 80 | 3 |
| 5 | MBConv6 k5 | 14² | 112 | 3 |
| 6 | MBConv6 k5 | 14² | 192 | 4 |
| 7 | MBConv6 k3 | 7² | 320 | 1 |
| head | Conv1×1 + pool + FC | 7² | 1280 | 1 |

All MBConv use SE ratio 0.25.

## Code

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def round_filters(channels, width_coeff, divisor=8):
    """Width scaling: scale by beta^phi, snap to a multiple of `divisor`,
    never dropping more than ~10% of the intended width."""
    if not width_coeff:
        return channels
    channels *= width_coeff
    new_c = max(divisor, int(channels + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * channels:
        new_c += divisor
    return int(new_c)


def round_repeats(num_layers, depth_coeff):
    """Depth scaling: scale per-stage repeats by alpha^phi, round up."""
    if not depth_coeff:
        return num_layers
    return int(math.ceil(depth_coeff * num_layers))


class MBConv(nn.Module):
    """Mobile inverted bottleneck: expand -> depthwise -> SE -> linear project,
    with a residual skip on the thin endpoints."""

    def __init__(self, in_ch, out_ch, kernel_size, stride,
                 expand_ratio=6, se_ratio=0.25):
        super().__init__()
        self.use_skip = (stride == 1 and in_ch == out_ch)
        mid = in_ch * expand_ratio

        layers = []
        if expand_ratio != 1:                       # 1x1 expand to the wide space
            layers += [nn.Conv2d(in_ch, mid, 1, bias=False),
                       nn.BatchNorm2d(mid), nn.SiLU()]
        layers += [nn.Conv2d(mid, mid, kernel_size, stride,  # kxk depthwise
                             padding=kernel_size // 2, groups=mid, bias=False),
                   nn.BatchNorm2d(mid), nn.SiLU()]
        self.expand_dw = nn.Sequential(*layers)

        se_dim = max(1, int(in_ch * se_ratio))      # squeeze-and-excitation
        self.se_reduce = nn.Conv2d(mid, se_dim, 1)
        self.se_expand = nn.Conv2d(se_dim, mid, 1)

        self.project = nn.Sequential(                # 1x1 project, LINEAR
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        h = self.expand_dw(x)
        s = F.adaptive_avg_pool2d(h, 1)
        s = self.se_expand(F.silu(self.se_reduce(s)))
        h = torch.sigmoid(s) * h                     # channel gate
        h = self.project(h)
        if self.use_skip:
            h = h + x
        return h


# Searched baseline (B0): (expand, kernel, stride, in, out, repeats)
BASE_STAGES = [
    (1, 3, 1,  32,  16, 1),
    (6, 3, 2,  16,  24, 2),
    (6, 5, 2,  24,  40, 2),
    (6, 3, 2,  40,  80, 3),
    (6, 5, 1,  80, 112, 3),
    (6, 5, 2, 112, 192, 4),
    (6, 3, 1, 192, 320, 1),
]


class EfficientNet(nn.Module):
    def __init__(self, width_coeff=1.0, depth_coeff=1.0,
                 dropout=0.2, num_classes=1000):
        super().__init__()
        stem = round_filters(32, width_coeff)
        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(stem), nn.SiLU())

        blocks, in_ch = [], stem
        for expand, k, stride, b_in, b_out, repeats in BASE_STAGES:
            out_ch = round_filters(b_out, width_coeff)
            for i in range(round_repeats(repeats, depth_coeff)):
                blocks.append(MBConv(in_ch, out_ch, k,
                                     stride if i == 0 else 1, expand))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        head = round_filters(1280, width_coeff)
        self.head = nn.Sequential(nn.Conv2d(in_ch, head, 1, bias=False),
                                  nn.BatchNorm2d(head), nn.SiLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(head, num_classes)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        x = self.pool(self.head(x)).flatten(1)
        return self.fc(self.dropout(x))


# Compound scaling: one knob phi -> (width, depth, resolution).
# The ratios come from a one-time grid search on B0 under alpha*beta^2*gamma^2 ~= 2:
#   alpha=1.2 (depth), beta=1.1 (width), gamma=1.15 (resolution).
# B1..B7 then follow d=alpha^phi, w=beta^phi, r=gamma^phi for phi=1..7, with the
# coefficients snapped to clean values / downsampling-friendly resolutions:
# (width_coeff, depth_coeff, resolution, dropout)
CONFIGS = {
    "b0": (1.0, 1.0, 224, 0.2),
    "b1": (1.0, 1.1, 240, 0.2),
    "b2": (1.1, 1.2, 260, 0.3),
    "b3": (1.2, 1.4, 300, 0.3),
    "b4": (1.4, 1.8, 380, 0.4),
    "b5": (1.6, 2.2, 456, 0.4),
    "b6": (1.8, 2.6, 528, 0.5),
    "b7": (2.0, 3.1, 600, 0.5),
}


def build(name="b0", num_classes=1000):
    width_coeff, depth_coeff, resolution, dropout = CONFIGS[name]
    model = EfficientNet(width_coeff, depth_coeff,
                         dropout=dropout, num_classes=num_classes)
    return model, resolution   # total FLOPS ~ (alpha*beta^2*gamma^2)^phi ~= 2^phi
```
