# AvgMax pooling (equal-weight mixed max-average global pooling), distilled

AvgMax is a parameter-free global pooling head that summarizes each convolutional feature map by
the **equal-weight blend of global average pooling (GAP) and global max pooling (GMP)**:
`f(x) = ½·max(x) + ½·avg(x) = (avg + max)/2`. It is the no-learning, fixed `a = 1/2` special case
of *mixed max-average pooling* `f_mix(x) = a·f_max(x) + (1-a)·f_avg(x)`, `a ∈ [0,1]`, which slides
along the average→max continuum. It keeps both complementary statistics — prevalence (average) and
peak (max) — in one number per channel, leaves the channel dimension `C` unchanged, adds no
parameters, and handles any spatial size.

## Problem it solves

Collapse a conv feature tensor `[B, C, H, W]` to a per-image channel vector `[B, C]` for
classification, with one fixed rule that must serve different backbones (channel counts `64`/`512`/
`1280`) and spatial sizes (`8×8` down to `1×1`), no label-dependent logic inside the layer, and
ideally no parameters (to avoid overfitting at this bottleneck). The difficulty: average pooling
and max pooling are each the better statistic only in part of the space, and the deciding regime
varies per channel and per dataset.

## Key idea and why it works

By a signal-to-noise (class-separability) analysis of a Bernoulli-activation feature model:

- **Average pooling** `f_a = (1/P)Σvᵢ` has mean `µ_a = α` (independent of pool size `P`) and
  variance `σ_a² = α(1-α)/P`. Mean-separation `|α₁-α₂|` is fixed, spread shrinks like `1/√P`, so
  its SNR **grows monotonically** with `P` — best for dense, broadly-firing channels and large
  pools. It estimates the activation *rate / prevalence* but divides any sparse peak by `P`.
- **Max pooling** `f_m = maxᵢvᵢ` has mean `µ_m = 1-(1-α)^P` and variance
  `σ_m² = (1-(1-α)^P)(1-α)^P`. Mean-separation `φ(P) = |(1-α₂)^P-(1-α₁)^P|` is **non-monotone** —
  `φ(1)=|α₁-α₂|`, and in the usual sparse-feature case it rises to an intermediate-cardinality peak
  (later for sparser features) before vanishing as both classes saturate at 1. In the sparse regime
  (`α₂ ≪ α₁ ≪ 1`) its SNR can **exceed** average pooling's. It keeps the *peak / strongest evidence*
  undiluted but reads one location and discards the other `N-1`.

**Neither dominates**; which wins depends on the channel's activation rate and the effective pool
size, both of which vary across channels, layers, architectures, and datasets — and cannot be
observed from inside the layer. The two summaries are lossy in opposite directions (average is
blind to the peak; max is blind to breadth/consistency), so they are complementary, not redundant.
The equal-weight blend keeps a contribution from each instead of committing to one endpoint.

**Why a fixed `a = 1/2`:** a convex linear blend interpolates the two endpoints (recovers GAP at
`a=0`, GMP at `a=1`), preserves units/scale, keeps the channel dim `C`, and is cleanly
differentiable. For the fixed no-learning variant, `a = 1/2` is the symmetric choice that gives no
prior preference to either statistic. The `/2` keeps the output at single-pooled magnitude so the
downstream classifier sees the usual scale.

**Gradient** (for `δ = ∂E/∂f`, a map of `N` locations):
`∂E/∂xᵢ = δ·(½·1[xᵢ=max] + ½/N)`. Every location is credited `δ·½/N` (the average branch, like
GAP) and the argmax gets an extra `δ·½` (the max branch, like GMP) — so training learns from both
prevalence and peak in one step.

## Final form

```
f_k = ½ · max_{x ∈ X_k} x  +  ½ · (1/N) Σ_{x ∈ X_k} x ,   N = H·W,   for each channel k.
```

## Working code

Fills the `CustomPool` slot of the global-pooling harness; size-agnostic adaptive reductions to
output 1, equal-weight blend, flatten to `[B, C]`; no parameters, no stability epsilon needed
(average and max of finite, ReLU-nonnegative activations are numerically benign).

```python
import torch.nn as nn
import torch.nn.functional as F


class CustomPool(nn.Module):
    """AvgMax pooling: equal-weight blend of global average and global max pooling.

    avg branch -> prevalence / mean activation rate (dense, broad channels)
    max branch -> peak / strongest evidence        (sparse, localized channels)
    Neither dominates across channels/datasets, so combine both at a = 1/2.
    Parameter-free; channel dim C unchanged; any spatial size H, W.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):                       # x: [B, C, H, W], activations >= 0 after ReLU
        avg = F.adaptive_avg_pool2d(x, 1)       # [B, C, 1, 1]  global average pooling
        mx = F.adaptive_max_pool2d(x, 1)        # [B, C, 1, 1]  global max pooling
        return ((avg + mx) / 2).view(x.size(0), -1)   # (avg + max)/2 -> [B, C]
```

## Relation to the pooling family

- **GAP** `(1/N)Σx` = this with `a = 0`.
- **GMP** `max x` = this with `a = 1`.
- **Mixed max-average pooling** `a·max + (1-a)·avg` with a **learnable** `a` (one per net / layer /
  channel / region) is the parametric generalization; trained via `∂f/∂a = max(x) - avg(x)`. AvgMax
  is its fixed equal-weight, parameter-free instance.
- **GeM (generalized mean)** `((1/N)Σxᵖ)^{1/p}` is a different one-parameter interpolation between
  average (`p=1`) and max (`p→∞`); it needs a stability epsilon for the power/root, AvgMax does not.
- Concatenating avg and max (rather than averaging) carries both too but doubles the channel
  dimension to `2C`; AvgMax's blend keeps `C` and the fixed classifier interface.
