# RMSNorm + Parallel Attention/MLP block, distilled

A transformer block design that makes two coupled changes to the standard pre-normalized GPT
block: it normalizes with **RMSNorm** instead of LayerNorm, and it computes attention and the
feed-forward sublayer **in parallel** from a single shared normalized input, summing both into
the residual. A cheaper normalization op inside a cheaper block structure.

## Problem it solves

In a deep, sharded decoder the transformer block is executed once per layer and replicated
across accelerators, so per-block costs — extra channel-wise reductions, extra
synchronization points, an unnecessary sequential dependency between sublayers — multiply by
the layer count and the device count and dominate training cost. The goal is a block that is
cheaper to run and cheaper to shard while leaving model quality intact, without changing the
data, optimizer, schedule, or the attention/feed-forward internals.

## Key idea

Two independent, composable moves.

**RMSNorm.** LayerNorm's two invariances are re-centering (from mean-subtraction) and
re-scaling (from dividing by the spread). Mean-subtraction shifts the activation vector but
does not change its variance or the variance of the gradients, so the scale control that
normalization relies on comes from the division, not the centering. Drop the mean and
normalize by the root-mean-square alone:

  ā_i = a_i / RMS(a) · g_i,   RMS(a) = sqrt( (1/n) Σ_{i=1}^n a_i^2 ),

with a learned per-channel gain g (init 1) and no bias (no mean is removed, so there is no
shift-invariance to restore). Because RMS is homogeneous under scalar rescaling,
RMS(αa) = |α|·RMS(a) and thus RMS(αa) = α·RMS(a) for positive α; positive re-scaling
invariance to inputs and to whole-weight-matrix scaling is preserved, while re-centering
invariance and per-weight-vector re-scaling invariance are given up. Cost per call: one reduction (sum of
squares) and a reciprocal-sqrt, versus LayerNorm's two reductions plus a subtraction. The
weight gradient is invariant to input scaling and inversely correlated with weight scale (an
implicit per-layer learning-rate adaptor).

**Parallel block.** The standard block is serial — the MLP reads the stream *after*
attention has been added:

  serial:   y = x + MLP(LN(x + Attn(LN(x))))   →   two norms, two residual rejoins, MLP waits for attention.

Both sublayers are additive perturbations of the same identity residual, so let them read one
shared normalized input and run independently, summing both into the residual:

  parallel: y = x + Attn(LN(x)) + MLP(LN(x)).

The residual stream stays an un-normalized identity path, so pre-normalization's well-behaved
gradients at initialization survive. One **shared (tied)** norm feeds both branches, which
halves the normalization layers per block. In a large-scale op-sharded implementation that
owns the attention and MLP kernels, the two branches read the same tensor, so the attention and
MLP input projections can be fused or co-scheduled, and the two sublayer outputs are summed
locally and rejoin the residual through a *single* all-reduce (one forward, one backward)
instead of two. That is the source of the roughly 15% large-scale throughput gain. In the local
decoder patch below, `CausalSelfAttention` and `MLP` are intentionally unchanged, so the code
realizes the single-norm parallel wiring but not a literal fused matmul rewrite. The only thing surrendered
is one step of composition (the MLP no longer conditions on attention's output): a small
quality cost at small scale that washes out by large scale. Tied vs untied (`MLP(LN2(x))` with
a second independent norm) makes no measurable quality difference, so the tied single-norm
form is chosen — it is the one that delivers the halved norm count and keeps the fused-kernel
path available.

## Defaults and why

- **eps = 1e-5**, inside the reciprocal-sqrt (`rsqrt(mean(x^2) + eps)`), floors a degenerate
  all-zero row.
- **No bias** on the norm: nothing was re-centered, so there is no shift to restore.
- **fp32 reduction**: the sum of squares over the channel dimension is computed in fp32 to
  preserve significance, then cast back to the working dtype.
- **One shared (tied) norm** per block: halves the norm count, preserves the fused-projection
  opportunity for kernel-owning implementations, and has quality equal to the untied variant in
  the checks I trust.
- **Residual-projection init** is not a forward change: a residual-output scheme on the order of
  `2/(L·sqrt(d))` keeps activations from growing with depth, the factor of two compensating for
  the parallel organization; the local scaffold already applies a `0.02/sqrt(2L)` scale to
  residual `c_proj.weight` parameters, matching two residual-producing projections per block.

## Working code

Drop-in for a GPT-style decoder (nanoGPT-style): keep the scaffold's `LayerNorm` class name and
`(ndim, bias)` signature, but replace its body with RMSNorm so the final norm changes too.
`Block` replaces the two-norm serial wiring with one-norm parallel wiring. `CausalSelfAttention`
and `MLP` are unchanged.

```python
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """RMSNorm under the scaffold's LayerNorm name. Normalize by RMS over the
    channel dimension; learned per-channel gain; no bias. The bias argument is
    intentionally ignored."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))   # gain g, init 1
        self.eps = 1e-5

    def forward(self, input):
        # rms_inv = 1 / sqrt(mean(x^2) + eps), reduction in fp32 for stability
        rms = input.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (input * rms).type_as(input) * self.weight   # x / RMS(x) * g


class Block(nn.Module):
    """Parallel attention + MLP block. One shared norm feeds both sublayers, which
    run on the same normalized input; their outputs are summed into the residual."""

    def __init__(self, config):
        super().__init__()
        self.ln = LayerNorm(config.n_embd, bias=config.bias) # single shared (tied) RMSNorm
        self.attn = CausalSelfAttention(config)              # unchanged
        self.mlp = MLP(config)                               # unchanged

    def forward(self, x):
        h = self.ln(x)                       # normalize once
        x = x + self.attn(h) + self.mlp(h)   # parallel branches, summed into the residual
        return x
```

## Relation to the prior block

- **vs LayerNorm:** drop the mean (re-centering), keep only RMS division (re-scaling). One
  reduction instead of two, no subtraction; equal to LayerNorm when the activations already
  have zero channel-mean.
- **vs the serial pre-LN block:** collapse two norms into one shared norm and the two
  sequential sublayers into two parallel branches summed into the residual; under op-sharding
  this halves the per-block all-reduces, and in systems that own the kernels it enables fused
  same-input projections. The local patch preserves the wiring while leaving attention/MLP
  internals unchanged.
