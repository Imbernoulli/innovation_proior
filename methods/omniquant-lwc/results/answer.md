# OmniQuant — Learnable Weight Clipping (LWC)

## Problem

Quantize an LLM's weights to 2/3/4 bits at a *PTQ budget* (small calibration set, short per-block local
optimization, no end-to-end backprop through the full model) while recovering near-QAT accuracy.
Min-max PTQ collapses at low bit-widths: the per-group clipping range is stretched to cover a few
outliers, so the handful of codes is spaced too coarsely for the bulk of the weights, and the rounding
is irreversible.

## Key idea

Make the per-group clipping range **learnable**, and parameterize it as a sigmoid-gated factor that can
only shrink the range inward (never expand past the min-max cover):

```
xmax = sigmoid(upbound_factor) * max(W_group)
xmin = sigmoid(lowbound_factor) * min(W_group)
```

with `upbound_factor`, `lowbound_factor` learnable per group (per side), initialized to `4.0` so
`sigmoid(4) ≈ 0.982` — the grid starts at the min-max cover (== plain PTQ) and *learns to clip inward*.
The scale/zero-point are computed from these clipped extremes; the fake-quant uses a straight-through
`round_ste` so the gradient reaches the factors through the differentiable scale.

## Why it works

- **Attacks the disease directly.** Clipping inward stops outliers from dictating the spacing; the codes
  pack onto the bulk and outliers clamp to the rail — the right move at INT2/INT3, chosen per group by
  the reconstruction loss instead of a heuristic.
- **Bounded, monotone, stable.** `sigmoid ∈ (0,1)` forbids expansion; init at the cover means it can
  only help (a clip-free group keeps its factor near init); the saturating gradient
  `sigmoid(γ)(1−sigmoid(γ))` damps the jagged range-loss at two bits.
- **PTQ budget.** The factors are *local* (a group's scale only affects its layer's output), so they are
  optimized block by block to minimize the block's output reconstruction MSE on ~128 calibration
  sequences — no global QAT.

In weight-*activation* settings, LWC is paired with a learnable equivalent transformation (channel-wise
scale/shift that migrates activation outliers into the weights), optimized in the same per-block loop.
For weight-only quantization, LWC is the complete weight-side mechanism.

## Scale / zero-point

Symmetric signed: `h = max(|xmax|, |xmin|) / (2^{N-1} − 1)`, clamped to `[1e-5, 1e4]`, `z = None`.
Asymmetric: `h = (xmax − xmin) / (2^N − 1)`, `z = −round(xmin / h)`.

## Code

```python
import torch, torch.nn as nn

CLIPMIN = 1e-5

def round_ste(x):
    return (x.round() - x).detach() + x          # round forward; identity backward

class UniformAffineQuantizer(nn.Module):
    def __init__(self, n_bits, group_size, shape, symmetric=False, lwc=False):
        super().__init__()
        self.n_bits = n_bits
        self.qmin, self.qmax = 0, 2 ** n_bits - 1
        self.group_size = group_size
        self.symmetric = symmetric
        self.lwc = lwc
        self.sigmoid = nn.Sigmoid()
        if lwc:
            init_value = 4.0                      # sigmoid(4) ~ 0.982: start at min-max cover
            self.upbound_factor = nn.Parameter(torch.ones(shape) * init_value)
            self.lowbound_factor = nn.Parameter(torch.ones(shape) * init_value)

    def calibration(self, x):
        xmax = x.amax(dim=-1, keepdim=True)
        xmin = x.amin(dim=-1, keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor) * xmax   # learnable inward clip (0,1)
            xmin = self.sigmoid(self.lowbound_factor) * xmin
        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = (abs_max / (2 ** (self.n_bits - 1) - 1)).clamp(min=CLIPMIN, max=1e4)
            zero_point = None
        else:
            scale = (xmax - xmin).clamp(min=CLIPMIN) / (2 ** self.n_bits - 1)
            zero_point = -(xmin / scale).round()
        return scale, zero_point

    def fake_quant(self, x, scale, zero_point):
        x_int = round_ste(x / scale)
        if zero_point is not None:
            x_int = x_int + zero_point
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dq = x_int
        if zero_point is not None:
            x_dq = x_dq - zero_point
        return x_dq * scale

    def forward(self, x):
        scale, zero_point = self.calibration(x)
        return self.fake_quant(x, scale, zero_point)
```
