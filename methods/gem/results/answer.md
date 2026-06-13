# GeM (Generalized-Mean) pooling, distilled

GeM pooling is a global spatial-pooling layer for CNN image descriptors. It collapses each
feature map `X_k` (its `N = W·H` non-negative activations, after a ReLU) to a single scalar via
the generalized (power) mean

```
f_k = ( (1/N) * sum_{x in X_k} x^p )^{1/p},
```

with one pooling exponent `p`. Average pooling (SPoC) is the special case `p = 1`; max pooling
(MAC) is the limit `p -> inf`. The exponent can be set by hand or, since the operation is
differentiable in `p`, learned by backpropagation — one shared scalar `p` (or one `p_k` per
channel). It is a drop-in replacement for global average/max pooling: input `[B, C, H, W]`, output
`[B, C]`, channel dimension unchanged, any spatial size.

## Problem it solves

A CNN descriptor is bottlenecked by the single spatial collapse from the conv feature tensor to a
fixed-length vector. The two standard collapses sit at opposite extremes — max keeps one location
per channel (discards the rest, brittle to a noisy peak), average weights every location equally
(dilutes a sparse strong response against a large dead field). Neither matches the sparse-strong
structure of real feature maps, and a linear blend `a*max + (1-a)*avg` can only interpolate
between two fixed summaries, not reshape how locations are weighted within the pool. GeM replaces
the fixed rule with one continuous, learnable selectivity dial.

## Key idea

Raise activations to a power `p` before averaging, then take the `p`-th root. For `p > 1` this
amplifies strong locations relative to weak ones before they are pooled, so the descriptor uses
*all* locations but weights the salient ones progressively harder as `p` grows — the graded
re-weighting a linear blend cannot express.

- **Endpoints.** `p = 1`: `((1/N) sum x)^1 =` arithmetic mean = SPoC. `p -> inf`: factor out
  `m = max_i x_i`, `M_p = m*((1/N) sum (x_i/m)^p)^{1/p}`; each `x_i/m in [0,1]` so its `p`-th
  power -> 0 except the maximizer(s) -> 1, the bracket -> `c in [1/N, 1]`, and `c^{1/p} -> 1`,
  hence `M_p -> m =` MAC.
- **Monotone dial.** By the power-mean inequality, `p < q => M_p <= M_q` (equality iff all `x_i`
  equal), and `M_p` is continuous in `p`. So `p >= 1` sweeps continuously and monotonically from
  the arithmetic mean up to the max — a single knob over selectivity.
- **Gradient = soft-argmax router.** With `S = (1/N) sum x^p` and `f = S^{1/p}` (so `S = f^p`),
  for positive values entering the power,
  `df/dx_i = (1/N) f^{1-p} x_i^{p-1}`. The backward weight on a location is
  `propto x_i^{p-1}`: uniform at `p=1` (average backward), concentrated on the max as `p -> inf`
  (max backward), and graded toward strong locations in between.
- **Learnable p.** `df/dp = (f/p^2)( ln(N / sum x^p) + p * sum(x^p ln x)/sum(x^p) )`, with `x`
  understood as the positive floored activations in code. It is built from the same sums as the
  forward, so `p` can be learned cheaply.

## Defaults and why

- **`p` initialized to `3.0`, learnable.** `p=1` is just the mean (no head start, flat
  location gradient); very large `p` starts at the brittle max. `p=3` is an intermediate
  contrast-enhanced regime that uses all locations while emphasizing the peaks.
- **`p` clamped to `>= 1`.** Keeps the operation in the avg-to-max regime, where the power mean is
  at least the arithmetic mean and emphasizes large values (power-mean inequality); `p < 1` bends
  toward geometric/harmonic/min behavior (wrong direction) and `p -> 0` is a singularity of the
  `(.)^{1/p}` form. The clamp also guards the learned parameter from drifting below 1.
- **`eps = 1e-6` floor before the power.** Activations are `>= 0` but include exact zeros; the
  power-mean calculus and `df/dp` term with `ln x` require positive values. Clamping the base to
  `>= eps` keeps `x^p`, `ln x`, and `x^{p-1}` finite for the values entering the power. The clamp
  itself has the usual saturated backward behavior below `eps`.
- **Single shared `p` vs per-channel `p_k`.** Per-channel is strictly more expressive but adds
  parameters and a harder, overfit-prone loss surface; a single shared learned `p` captures most
  of the benefit with one clean dial. Per-channel is the available generalization.

## Final form

```
GeM(X)_k = ( (1/|X_k|) * sum_{x in X_k} clamp(x, eps)^p )^{1/p},   p >= 1,  eps = 1e-6,
```
with `p` a (shared) learnable parameter initialized to `3.0`. Implemented by reusing the global
average-pool primitive: raise to `p`, average over the whole `H x W` grid, raise to `1/p`. This is
size-agnostic (any `H, W`, including `1x1`) and leaves the channel dimension `C` unchanged.

## Working code

Filling the global-pooling slot (`[B, C, H, W] -> [B, C]`):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Generalized-mean pooling: f_k = ((1/N) sum_x x^p)^{1/p} per channel.
    p=1 -> average pooling; p->inf -> max pooling. p is a shared learnable dial."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)   # shared selectivity dial, init 3.0
        self.eps = eps                             # positivity floor

    def forward(self, x):                          # x: [B, C, H, W], x >= 0 after ReLU
        p = self.p.clamp(min=1.0)                  # stay in the avg..max regime
        x = x.clamp(min=self.eps)                  # strictly positive base
        # x^p, average over the whole H x W grid (= (1/N) sum), then ^(1/p)
        return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p).view(x.size(0), -1)

    def __repr__(self):
        return f"{type(self).__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"
```

Equivalent functional form (matching the standard implementation):

```python
def gem(x, p=3, eps=1e-6):
    # raise to p, global-average over H x W, take the p-th root
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)
```

Per-channel variant: make `p` a length-`C` parameter and broadcast it over the spatial dims
(`self.p` of shape `[C]`, reshaped to `[1, C, 1, 1]` before `x.pow(p)`); same forward otherwise.
