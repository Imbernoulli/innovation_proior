# Sandwich LayerNorm (Sandwich-LN), with RMSNorm, distilled

Sandwich-LN is a transformer block design that wraps each sublayer (attention, MLP) in a
normalization on **both** sides: a pre-norm before the sublayer and a post-norm on the sublayer's
**branch output, before it is added back to the residual stream**. The residual path itself stays
an un-normalized identity skip. Combined here with **RMSNorm** as the normalization op, the block
keeps pre-normalization's well-behaved-at-initialization gradients while bounding the forward
activation scale that an un-normalized residual path lets grow with depth.

## Problem it solves

A deep, pre-normalized transformer trained at scale in 16-bit precision overflows: the residual
stream is an un-normalized identity path, so every sublayer's raw output accumulates on it, the
forward scale grows with depth, and in a few persistent outlier dimensions it compounds into the
`10^4–10^5` range, past the FP16 ceiling, producing NaN losses within hundreds of iterations.
Post-LN (normalize the residual sum) bounds the forward scale but reintroduces the large,
depth-independent output-layer gradients that make Post-LN require warmup and train delicately.
The goal is a block that keeps the forward scale bounded with depth **without** giving up
pre-normalization's gradient health, leaving the data, optimizer, schedule, and the
attention/MLP internals untouched.

## Key idea

The un-normalized residual path causes both the good gradients (a clean skip) and the forward
explosion (raw branch outputs accumulate). Do not normalize the path; normalize the **branch
output** just before it rejoins the path:

  x ← x + Norm( Sublayer( Norm(x) ) ).

A normalized branch term has fixed per-coordinate variance `β₀` regardless of the sublayer's raw
output magnitude, so the residual recursion changes from compounding,
`Var(x_{l+1}) ≈ Var(x_l)·(1 + …)` (worst-case geometric growth, because branch outputs correlate
with the directions the stream is already large in), to additive,
`Var(x_{l+1}) ≈ Var(x_l) + β₀` (bounded, linear in depth). Because the norm is on the branch and
**not** on the residual sum, the identity skip is still un-normalized: gradients flow back through
it unimpeded, so the well-behaved-at-init gradient property of Pre-LN is preserved (this is the
distinction from Post-LN, which normalizes the sum). The branch-end norm additionally damps the
backward pass by `~1/‖a‖`, keeping the gradient through a misbehaving branch bounded.

Each sublayer is therefore "sandwiched": a pre-norm (input/gradient health) and a post-norm on
the branch output (forward-scale control). The pre-norm earns the gradient property; the post-norm
earns the scale bound; placing the post-norm on the branch rather than the sum means it does not
cost the gradient property.

## The normalization op: RMSNorm

With four norm calls per block, the per-call cost matters. LayerNorm subtracts the mean
(re-centering) and divides by the standard deviation (re-scaling); mean-subtraction shifts the
activations but changes neither their variance nor the gradients' variance, so the scale control
comes from the division. Drop the mean and normalize by the root mean square alone:

  ā_i = a_i / RMS(a) · g_i,   RMS(a) = sqrt( (1/n) Σ_{i=1}^n a_i² ),

learned gain `g` (init 1), no bias (no mean removed → no shift to restore). It equals LayerNorm
when the input mean is zero. `RMS(αa) = α·RMS(a)` keeps re-scaling invariance to inputs and to
the whole weight matrix; re-centering and per-weight-vector re-scaling invariance are given up.
The weight gradient is invariant to input scaling and inversely correlated with weight scale (an
implicit per-layer learning-rate adaptor). Cost: one reduction (sum of squares) + a
reciprocal-sqrt, versus LayerNorm's two reductions + a subtraction.

## Defaults and why

- **eps = 1e-5**, inside the reciprocal-sqrt (`rsqrt(mean(x²) + eps)`): floors a degenerate
  all-zero row; sized below any healthy activation RMS so it does not perturb the normal regime.
- **No bias** on the norm: nothing is re-centered, so there is no shift to restore.
- **fp32 reduction**: the sum/mean of squares over the channel dimension is computed in fp32
  (the squaring + wide sum is exactly what loses significance or overflows in fp16), then cast
  back to the working dtype — the stability the method is for must not be undone inside the norm.
- **gain init 1**: the norms are scale-identity at initialization, so the carefully-set initial
  forward map (and convergence) is essentially unchanged; the post-norms act on a well-behaved
  branch output at init and only cap magnitude during training.

## Final block

```
pre-norm  ->  attention  ->  post-norm  ->  add to residual
pre-norm  ->  MLP        ->  post-norm  ->  add to residual
```

i.e. `x = x + ln_post1(attn(ln_pre1(x)))`, then `x = x + ln_post2(mlp(ln_pre2(x)))`. Contrast:
Pre-LN is `x = x + Sublayer(LN(x))` (one norm, before); Post-LN is `x = LN(x + Sublayer(x))`
(one norm, on the sum); Sandwich-LN is `x = x + Norm(Sublayer(Norm(x)))` (two norms, the
sublayer in the middle, identity path un-normalized).

## Working code

Drop-in for a GPT-style decoder (nanoGPT-style). The norm keeps the `(ndim, bias)` signature so
it replaces the existing norm class unchanged; the block gains a post-norm per sublayer.
`CausalSelfAttention` and `MLP` are unchanged.

```python
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (drop-in for the (ndim, bias) norm slot)."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))   # gain g (init 1); no bias
        self.eps = 1e-5

    def forward(self, input):
        # fp32 reduction for fp16-safe sum of squares; 1 / sqrt(mean(x^2) + eps)
        rms = input.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (input * rms).type_as(input) * self.weight


class Block(nn.Module):
    """Sandwich-LN block: each sublayer wrapped by a pre-norm and a post-norm on the
    branch output, added to an un-normalized residual stream."""

    def __init__(self, config):
        super().__init__()
        self.ln_pre1  = RMSNorm(config.n_embd, bias=config.bias)   # before attention
        self.attn     = CausalSelfAttention(config)
        self.ln_post1 = RMSNorm(config.n_embd, bias=config.bias)   # on attention's branch output
        self.ln_pre2  = RMSNorm(config.n_embd, bias=config.bias)   # before MLP
        self.mlp      = MLP(config)
        self.ln_post2 = RMSNorm(config.n_embd, bias=config.bias)   # on the MLP's branch output

    def forward(self, x):
        x = x + self.ln_post1(self.attn(self.ln_pre1(x)))   # x + Norm(Attn(Norm(x)))
        x = x + self.ln_post2(self.mlp(self.ln_pre2(x)))    # x + Norm(MLP (Norm(x)))
        return x
```

Everything else in the decoder — token/position embeddings, the block stack, the final norm, the
tied output head, the optimizer, the learning-rate schedule, and the data pipeline — is
unchanged. The method touches only the normalization op and the block's internal wiring.
