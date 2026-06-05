# RMSNorm (Root Mean Square Layer Normalization)

## Problem

Layer normalization stabilizes training and speeds up convergence (in steps), but it runs at
every layer — and in RNNs at every timestep — and each call computes two statistics (a mean and
a standard deviation) plus an elementwise mean-subtraction. This per-step overhead erodes the
net speedup, badly so in deep and recurrent models. RMSNorm asks which part of the standard
layer is actually responsible for the benefit, and removes the rest.

## Key idea

The standard layer does two things: it subtracts the mean (giving **re-centering** invariance)
and divides by the standard deviation (giving **re-scaling** invariance). Mean-subtraction
recenters the activations but does not change their variance, so it is not the direct
spread-control term for activations or gradients. The hypothesis: **re-scaling invariance is what
makes layer normalization work; re-centering is dispensable.** Drop the mean-subtraction and
normalize only by the root mean square.

## Method

For summed inputs a ∈ R^n to a layer:

  āᵢ = aᵢ / RMS(a) · gᵢ,   RMS(a) = √( (1/n) Σ_{i=1}^n aᵢ² ),

with a learned per-neuron gain g (init 1) and no bias by default (no mean is removed, so nothing
needs restoring). RMS forces a onto the sphere of radius √n. When the mean of a is already zero,
RMSNorm equals the standard mean-subtracting layer.

**Invariances.** Because RMS is linear, RMS(αa) = α·RMS(a), so RMSNorm is invariant to re-scaling
of the inputs, the dataset, and the whole weight matrix (the α cancels), and to single-case
re-scaling. It is **not** invariant to per-weight-vector re-scaling (RMS pools all neurons), and
**not** to re-centering / shifts (there is no mean to absorb them) — the one property it gives up
relative to the standard layer.

| | W re-scale | W re-center | weight-vec re-scale | data re-scale | data re-center | single-case re-scale |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| BatchNorm | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ |
| WeightNorm | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| LayerNorm | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ |
| RMSNorm | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ |
| pRMSNorm | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ |

**Gradients.** With v the pre-nonlinearity expression v = (Wx)/RMS(a) ⊙ g + b:
∂L/∂b = ∂L/∂v and ∂L/∂g = ∂L/∂v ⊙ (Wx)/RMS(a) — both invariant to scaling x and W, and the
g-gradient is proportional to the *normalized* inputs, keeping g's scale stable. For the weights,
let u = ∂L/∂v and

  R = (1/RMS(a))·( I − (Wx)(Wx)ᵀ/(n·RMS(a)²) ),   ∂L/∂W = (R(g ⊙ u)) xᵀ.

Scaling x or W by δ sends R → R/δ. Combined with the explicit x in the gradient, ∂L/∂W is
**invariant to input scaling** (the δ from x cancels the 1/δ from R) and **inversely proportional
to weight scaling** (only R moves), so larger weights receive smaller gradients — an implicit,
per-layer learning-rate adaptation that keeps weight norms in check.

**Efficiency.** One reduction (sum of squares) and no subtraction, versus two reductions and a
subtraction. The saving is most valuable where the normalization is executed many times per
training step, especially recurrent models with a normalization at each timestep.

## pRMSNorm (partial)

Assuming the n neurons are roughly i.i.d., estimate the RMS from the first k entries selected by a
fraction p of the layer and normalize all n by it. In the mathematical description k = ceil(n·p);
the implementation below follows the integer slice used in code:

  RMS̄(a) = √( (1/k) Σ_{i=1}^k aᵢ² ),  āᵢ = aᵢ / RMS̄(a) · gᵢ.

Linearity still holds, so pRMSNorm has the same invariances as RMSNorm while touching only part
of the vector. RMS̄ is a noisy and generally biased estimate of the full-vector RMS; gradients can
explode for very small k, so p is a cost-stability knob rather than a free reduction.

## Code

```python
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
        d:    layer width (number of summed inputs / neurons).
        p:    partial ratio; this implementation treats 0 <= p <= 1 as partial,
              so practical partial use should keep p > 0. p < 0 or p > 1 disables partial mode.
        eps:  numerical floor on the denominator.
        bias: optional shift; off by default since RMSNorm does no mean-subtraction
              and therefore enforces no re-centering invariance to restore.
        """
        super().__init__()
        self.eps, self.d, self.p, self.bias = eps, d, p, bias
        self.scale = nn.Parameter(torch.ones(d))          # gain g, init 1
        self.register_parameter("scale", self.scale)
        if bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)      # ||a|| = sqrt(sum a_i^2): one reduction
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)           # use first int(d*p) entries
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)                 # full RMS or partial estimate = selected norm / sqrt(d_x)
        x_normed = x / (rms_x + self.eps)                 # a_i / RMS(a), no mean subtracted

        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed                      # a_i / RMS(a) * g_i
```
