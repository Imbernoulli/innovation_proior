# Context: the Transformer feed-forward nonlinearity (circa 2020-2021)

## Research question

A Transformer decoder block alternates two sublayers: multi-head self-attention and a
position-wise feed-forward network (FFN). The FFN is applied identically and independently at
every sequence position and carries roughly two-thirds of the block's parameters and a large
share of its FLOPs. Its structure is fixed and minimal — project up to a wide hidden
dimension, apply a pointwise nonlinearity, project back down:

```
FFN(x) = act(x W1 + b1) W2 + b2,    W1: d -> d_ff,  W2: d_ff -> d,   d_ff = 4d.
```

Almost all of the design attention in this layer has gone to `W1`, `W2`, and `d_ff`; the
single scalar nonlinearity `act` is usually inherited unquestioned from whatever the codebase
shipped with (ReLU in the original Transformer, GELU in the BERT/GPT line). The precise goal
here is to find a *better* `act` for autoregressive language-model pretraining: a change that
lowers validation loss / improves sample efficiency per unit of training compute, while
staying a strictly **modular, feed-forward-only** intervention. To be adoptable it must
(1) keep the `(B, T, d) -> (B, T, d)` shape contract of the FFN; (2) not touch attention,
normalization, the dataset, the optimizer schedule, or evaluation; (3) ideally add no
parameters and require no re-tuning, so practitioners can drop it into an existing codebase.
A solution that needs a bigger hidden state, a third weight matrix, or a new hyperparameter
pays a cost that has to be earned back. Closing the gap between "the nonlinearity is whatever
we inherited" and "the nonlinearity is the best available drop-in" is the problem.

## Background

By this period the position-wise FFN is the least-examined part of the Transformer. The field
state and the load-bearing concepts:

- **The FFN as a fixed two-matrix sandwich.** The original Transformer (Vaswani et al. 2017)
  defines `FFN(x) = max(0, x W1 + b1) W2 + b2`, with `d_ff = 4d`. In the T5 lineage (Raffel et
  al. 2019) the biases are dropped, giving `FFN_ReLU(x) = max(x W1, 0) W2`. The nonlinearity
  is a single pointwise function applied between the two projections; it is the only nonlinear
  element in the sublayer.

- **Pointwise activations are treated as interchangeable, smooth, ~linear-above-threshold.**
  The activations that have displaced ReLU in large language models — GELU and Swish (below) —
  share a shape: smooth, with a small nonmonotonic dip just below zero, and asymptotically
  *linear* for large positive input (GELU(x) -> x and Swish(x) -> x as x -> +inf). The tacit
  assumption in the field is that a good activation passes large positive signals through
  roughly unchanged and differs from ReLU mainly in how gently it handles the region around
  zero.

- **Rectified polynomials as activations, from associative-memory theory.** Krotov & Hopfield
  (2016) study dense associative memories whose energy uses a rectified polynomial
  `F(x) = x^n` for `x >= 0` and `0` for `x < 0`, with integer `n`. The `n = 2` case recovers
  the standard quadratic Hopfield network; for `n > 2` each term in the energy becomes
  *sharper*, which lets the network pack and reliably retrieve more memories — they derive a
  capacity `K^max = alpha_n N^{n-1}` that grows with `n`. They state the open question
  directly: above the threshold, should an activation grow linearly, sub-linearly, or *faster*
  than linearly, and are there functions that beat ReLU? This is the precedent that a
  rectified, faster-than-linear activation can have a principled upside — but it lives in
  shallow energy-based models and on MNIST, never in a deep Transformer FFN.

- **Multiplicative interactions as a source of representational power.** A separate line
  (Jayakumar et al. 2020) shows that layers built on a *product* of two learned quantities
  (gating, attention, hypernetworks, bilinear forms) strictly enrich the class of functions a
  network can represent compared to purely additive layers of the same size. A multiplicative
  (degree-2) term is a cheap way to buy expressivity that a linear-plus-pointwise stack lacks.

- **Searching for architecture components pays off, but at a coarse grain.** Prior automated
  searches over neural-net components (e.g. activation-function search, below) had shown that
  a learned or searched primitive can beat the hand-chosen default. But these searches fixed
  the surrounding structure and varied one thing — a scalar function, or a high-level block —
  so the candidate space was narrow.

## Baselines

The prior FFN-nonlinearity choices a new activation would be measured against.

- **ReLU FFN (Vaswani et al. 2017; Glorot et al. 2011).** `act(z) = max(0, z)`. Cheap,
  nonsaturating on the positive side, induces sparsity (negative pre-activations are zeroed).
  Core idea: pass the positive part linearly, kill the rest. **Limitation:** the positive
  branch is exactly the identity, so the FFN's only nonlinearity is the hard gate at zero;
  the function is piecewise *linear* and treats every surviving unit on the same linear scale,
  with a hard nondifferentiable kink at the origin.

- **GELU (Hendrycks & Gimpel 2016).** `act(z) = z Phi(z) = z * 1/2 [1 + erf(z / sqrt(2))]`,
  with the common approximations `0.5 z (1 + tanh(sqrt(2/pi)(z + 0.044715 z^3)))` and
  `z sigmoid(1.702 z)`. Derived as the expected value of an input multiplied by a
  Bernoulli(Phi(z)) "keep" mask — a deterministic stand-in for input-dependent (adaptive)
  dropout. Smooth everywhere, with a small negative dip just below zero. The default
  activation in BERT and the GPT line. **Limitation:** it is a smooth reshaping of ReLU that
  is still asymptotically linear (`Phi(z) -> 1`, so `GELU(z) -> z`), so for strongly activated
  units it behaves essentially like the identity; it also costs an `erf`/`exp`/`tanh`
  evaluation per element.

- **Swish (Ramachandran, Zoph & Le 2017).** `act(z) = z * sigmoid(beta z)`, the winner of a
  reinforcement-learning-plus-exhaustive *search over scalar activation functions*. Self-gated
  (the input gates itself through a sigmoid of itself), smooth, nonmonotonic, and — like
  GELU — asymptotically linear for large positive input. Demonstrates that searching for the
  nonlinearity beats hand-design. **Limitation:** the search ranged over *scalar pointwise
  functions* with the FFN's two-matrix structure held fixed, and the discovered functions all
  land in the same smooth, ~linear-asymptote family as GELU; the search never reached
  functions with a different growth regime or a multiplicative structure.

- **GLU and its Transformer variants (Dauphin et al. 2016/2017; Shazeer 2020).** A Gated
  Linear Unit replaces a single projection with a *product of two* projections, one passed
  through a nonlinearity that gates the other: `GLU(x) = (x W) ⊗ sigmoid(x V)`, with a
  "bilinear" variant `(x W) ⊗ (x V)` that drops the sigmoid. Dauphin et al.'s argument for
  gating is gradient flow: the product carries a linear path `x ⊗ sigma(x)` whose gradient
  term `∇x ⊗ sigma(x)` is *not* downscaled by a saturating derivative, unlike `tanh`-style
  gating which vanishes through depth. Shazeer (2020) drops these into the Transformer FFN:
  `FFN_ReGLU(x) = (max(0, x W) ⊗ x V) W2`, and likewise `GEGLU` (GELU gate) and `SwiGLU`
  (Swish gate). On a T5 span-filling benchmark `GEGLU` and `SwiGLU` give the best perplexities
  of the family. **Limitation:** every GLU FFN uses *three* weight matrices instead of two, so
  to hold parameters and compute fixed the hidden width `d_ff` must be cut by a factor of 2/3;
  it adds a projection and a hyperparameter (the width adjustment), and Shazeer (2020)
  explicitly declines to explain *why* the gating helps — the mechanism is left open.

## Evaluation settings

The yardsticks already in use for this kind of FFN change, as pre-existing facts:

- **Search/diagnostic task:** autoregressive (decoder-only) language modeling on the One
  Billion Words Benchmark (LM1B), short sequences (length 64), ~35M-parameter models, fixed
  training-compute budget; fitness measured by validation perplexity at a fixed budget rather
  than by step time. A "speedup factor" is read off as the fraction of a vanilla baseline's
  compute needed to reach the baseline's final quality.

- **Transfer / scale-up:** decoder-only LM on larger corpora (C4, PG19) and across codebases
  (Tensor2Tensor, T5, Lingvo), at model sizes from ~20M to ~2B parameters, using each
  codebase's *default* Transformer hyperparameters with regularization disabled and no
  re-tuning. Optimizer is Adafactor with 10K warmup at lr 0.01 and inverse-square-root decay.

- **Metrics:** cross-entropy validation loss / perplexity (primary, lower better); for the
  larger comparisons, downstream one-shot task accuracy in a GPT-3-style pretraining-then-
  one-shot protocol. The target task for this trace's harness is GPT-2-Medium-scale
  (24 layers, d=1024, ~355M params) decoder-only pretraining on FineWeb-10B with the GPT-2
  tokenizer, scored by FineWeb validation cross-entropy and by WikiText-2 / LAMBADA perplexity
  and ARC-Easy / HellaSwag / PIQA / WinoGrande accuracy.

## Code framework

The intervention plugs into the standard GPT-style decoder block: each block is
`x = x + attn(norm(x)); x = x + ffn(norm(x))`, and the FFN is the two-matrix sandwich. Only
the FFN sublayer is in scope here, and within it only the nonlinearity between the two
projections is unsettled — that pointwise map is exactly the slot to fill. Everything around
it (the up-projection `c_fc`, the down-projection `c_proj`, the 4x hidden width, dropout, the
attention/normalization/optimizer/data pipeline) already exists and is held fixed.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Position-wise feed-forward sublayer of a Transformer decoder block.
    Up-project to 4x width, apply a pointwise nonlinearity, project back down.
    Must map (B, T, n_embd) -> (B, T, n_embd) and add no dependence on
    attention/normalization/data/optimizer."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)            # up-projection to the 4x hidden dimension
        # TODO: the pointwise nonlinearity we will choose for this hidden state
        x = self.c_proj(x)          # down-projection back to n_embd
        x = self.dropout(x)
        return x


# existing decoder block the FFN plugs into (attention / norm already defined)
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)   # fixed: not modified here
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))            # the FFN sublayer being designed
        return x
```

The up- and down-projections and the width are given; the single empty slot is the pointwise
map applied to the hidden state between them.
