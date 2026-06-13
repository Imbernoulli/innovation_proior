# Context: the transformer block under depth and precision pressure (circa 2020-2021)

## Research question

A GPT-style decoder is a stack of identical transformer blocks; the same block runs once per
layer, and the models being trained are getting both wider and much deeper, dozens to over a
hundred layers, often trained in 16-bit precision to fit memory and to run fast. Inside each
block two regions are not fixed by anything fundamental: the **normalization rule** (how
activations are rescaled around each sublayer) and the **block structure** (how the attention
sublayer, the feed-forward sublayer, the normalization, and the residual connection are wired
together). The default is LayerNorm with a learned scale and bias, in a pre-normalization
arrangement: `x + Attn(LN(x))`, then `x + MLP(LN(x))`. That default keeps the gradients
well-behaved at initialization, which is why it displaced the original post-normalization
arrangement. But the precise question here is what happens to the *forward* activations as the
same block is stacked very deep: the residual path of a pre-normalized block is an
un-normalized identity highway onto which every sublayer adds its raw output, and nothing in the
block bounds how large the values on that highway can become.

The goal is a block whose normalization and wiring reduce the validation loss reached for a
fixed compute budget on deep, large-scale pretraining — and that does so *robustly*, without the
training diverging — while leaving the dataset, tokenizer, optimizer, schedule, and the
attention and feed-forward computations themselves untouched. Two sub-questions sit underneath.
First, which part of the standard normalization is actually responsible for its stabilizing
effect, so the rest can be removed. Second, given that pre-normalization is chosen specifically
to keep the residual path un-normalized (for the gradient benefit), what controls the *scale*
of the activations flowing along that un-normalized path as depth grows, and what would have to
change in the block to keep that scale bounded without giving up the gradient benefit. A good
answer keeps the well-behaved-at-initialization gradients of pre-normalization while preventing
the forward activations from growing without bound through depth.

## Background

**Why normalization is in the block at all.** Normalization layers stabilize and speed up
training (measured in steps) by keeping the scale of activations and gradients under control as
signals pass through a deep stack. LayerNorm (Ba, Kiros & Hinton 2016) computes, for the summed
inputs `a ∈ R^n` to a layer, a mean and a standard deviation across the `n` channels and
standardizes: `ā_i = (a_i − μ)/σ · g_i (+ b_i)`, with `μ = (1/n)Σ a_i`,
`σ = sqrt((1/n)Σ(a_i − μ)^2)`, a learned per-channel gain `g` (init 1) and optional bias `b`.
It uses no cross-example statistics, so it handles variable-length sequences and is the de-facto
normalizer in transformers. A standard analysis attributes its benefit to two invariances:
**re-centering** invariance (output unchanged if inputs or weights are shifted by a constant)
from the mean-subtraction, and **re-scaling** invariance (output unchanged if inputs or weights
are scaled) from the division by `σ`. A useful fact about its output magnitude: the standardized
vector has unit per-channel variance, so before the learned gain its `L2` norm is on the order
of `√n` — for a model width of a couple of thousand, that is a norm of order tens.

**Pre-LN vs Post-LN — where the norm sits, and what each one does to gradients and to the
forward scale.** The original transformer placed normalization *after* the residual addition
(Post-LN): `x ← LN(x + Sublayer(x))`. Xiong et al. (2020) analyzed both placements at
initialization with mean-field arguments. For Post-LN, the expected gradient norm of the
parameters near the output is `O(d√(ln d))`, *independent of depth* `L` — large, which is why
Post-LN needs a learning-rate warmup stage and is delicate to train. Placing the norm *inside*
the residual branch (Pre-LN), `x ← x + Sublayer(LN(x))`, so the residual stream itself is never
normalized, makes the last-layer gradient `O(d√(ln d / L))` — it shrinks with depth, is
well-behaved at initialization, and trains stably without warmup. That gradient benefit is the
reason Pre-LN is the prevailing choice. The same analysis records a *forward*-side fact that is
the load-bearing one here: in Post-LN the expected squared hidden-state norm is constant in
depth, `E[‖x^post_{l}‖^2] = (3/2) d`, whereas in Pre-LN it **grows linearly with depth**,
`(1 + l/2) d ≤ E[‖x^pre_{l}‖^2] ≤ (1 + 3l/2) d`. The mechanism is structural: the residual
stream is an identity path, so each layer's sublayer output is added in raw and never rescaled,
and the norms accumulate down the stack.

**The forward-scale failure mode in deep, low-precision training.** Pretraining large models
(billions of parameters) commonly runs in 16-bit precision to save memory and speed up compute;
some frameworks store parameters and gradients in FP16 throughout. FP16 has a narrow dynamic
range, so an activation that grows past roughly `6.5 × 10^4` overflows to infinity, and any
statistic computed from it — in particular the variance inside a normalization layer — becomes
NaN. Two failure modes are distinguished in this regime: **overflow** (NaN losses, from values
too large) and **underflow** (diverging loss, from gradients too small to represent). The
overflow mode interacts directly with the un-normalized residual path above: because the
pre-normalized residual stream's scale grows with depth and is never rescaled, the largest
activation dimensions can climb, layer after layer, into the range where the next layer's
normalization overflows. A concrete diagnostic of this: in a deliberately stressed setting —
deep stack, large learning rate, small batch — a pre-normalized transformer's maximum
activation value in the final embeddings explodes within a few hundred iterations into the
`10^4–10^5` range and the training NaNs out, whereas the constant-scale Post-LN arrangement does
not exhibit the same runaway growth. So the very property that makes pre-normalization good for
gradients (the un-normalized identity path) is what leaves its forward activations unbounded
with depth.

**Why specific dimensions explode rather than the whole vector uniformly.** The output of a
normalization layer is, dimension by dimension, the standardized input times a learned gain, so
its overall scale is on the order of `√d`. Transformers are known to carry a few activation
dimensions that are persistently much larger than the rest; when the normalized input is fed
through a sublayer's projections, those large directions are reproduced in the sublayer's output
at scale `10^1–10^2`. Added back onto the residual highway, they make the *next* layer's input
larger in exactly those dimensions, which makes that layer's sublayer output larger still — a
layer-by-layer aggravation. This is an analytic consequence of (i) the `√d` output scale of
normalization, (ii) the persistent large-dimension structure of transformer activations, and
(iii) the un-normalized residual add that re-injects each branch's output into the next layer.

**A diagnostic finding about the standard normalization op itself.** Its stabilizing effect is
usually credited to its two invariances, but mean-subtraction does not reduce the variance of
the hidden states or of the gradients — it only shifts them. This raises the possibility that
the **re-scaling** invariance (the division by the spread) is what does the stabilizing work and
that the **re-centering** invariance (the mean-subtraction) is dispensable; if so, one of the
two channel-wise reductions and the subtraction could be removed at no quality cost. The
Euclidean norm (without the `1/n` factor) had been used for weight normalization (Salimans &
Kingma 2016) but had not been made to work as a layer-activation normalizer.

## Baselines

**Standard pre-normalized LayerNorm block (Vaswani et al. 2017; Pre-LN per Xiong et al. 2020).**
The default. Two LayerNorms and two residual additions per block, in series:

```
x ← x + Attn(LN1(x))
x ← x + MLP(LN2(x))
```

The residual path is an identity highway and each sublayer reads a freshly normalized copy of
it. **Limitation:** the gradients are well-behaved at initialization, but the residual stream is
never rescaled, so its forward scale grows with depth (`E[‖x^pre_l‖^2]` linear in `l`) and the
largest activation dimensions are re-injected and magnified layer after layer; in a deep stack
trained at scale in 16-bit precision, those dimensions can climb until a normalization overflows
and the loss goes NaN. The block has no mechanism to bound the magnitude of what flows along its
identity path.

**Post-normalized LayerNorm block (Vaswani et al. 2017, original placement).** Normalize *after*
the residual addition:

```
x ← LN1(x + Attn(x))
x ← LN2(x + MLP(x))
```

Because the norm sits on the residual sum, the forward hidden-state scale is held constant in
depth (`E[‖x^post_l‖^2] = (3/2) d`), so it does not suffer the runaway forward growth above.
**Limitation:** normalizing the residual sum is exactly what makes the parameter gradients near
the output large at initialization (`O(d√(ln d))`, independent of `L`); the arrangement requires
a carefully tuned learning-rate warmup and is delicate and slow to optimize, especially as depth
grows. It buys forward-scale control at the price of the gradient health that motivated
pre-normalization in the first place.

**RMSNorm (Zhang & Sennrich 2019).** Tests the hypothesis that re-centering is dispensable by
dropping the mean entirely and normalizing by the root-mean-square alone:

```
ā_i = a_i / RMS(a) · g_i,   RMS(a) = sqrt( (1/n) Σ_{i=1}^n a_i^2 )
```

with learned gain `g` (init 1) and no bias by default — there is no mean removed, hence no
re-centering invariance a bias would need to restore. Because RMS is linear,
`RMS(αa) = α·RMS(a)`, RMSNorm is invariant to re-scaling of the inputs and of the whole weight
matrix (the `α` cancels) but is **not** invariant to shifts and **not** to per-weight-vector
re-scaling. It does one reduction (sum of squares) and no subtraction, versus LayerNorm's two
reductions and a subtraction, and equals LayerNorm exactly when the input's mean is already
zero. Its weight gradient is invariant to input scaling and inversely correlated with weight
scale, acting as an implicit per-layer learning-rate adaptor. **Where it leaves off:** it is a
drop-in replacement for the *normalization op only*. It changes neither where the norm sits nor
how the residual and sublayers are wired, so on its own it does nothing about the forward-scale
growth of a pre-normalized stack. It is an ingredient, not a block design.

**Initialization-based residual scaling (Zhang, Dauphin & Ma 2019, "Fixup").** A different
route to deep stability: with sufficiently careful per-layer initialization, residual networks
can be trained without any normalization. **Limitation:** it works by rescaling many
initialization layers in a model-specific way and is awkward to transfer between architectures;
it does not give a drop-in block change, and in models that keep normalization for its other
benefits it does not address the scale aggravation along an un-normalized residual path.

## Evaluation settings

The natural yardstick is GPT-style decoder-only pretraining held fixed except for the block's
normalization and wiring. A representative protocol: a GPT-2-Medium-class model (24 layers, 16
heads, model width 1024, ≈355M parameters), pre-normalized decoder, trained on a large web-text
corpus (a FineWeb-scale 10B-token sample) with the GPT-2 byte-pair tokenizer over a few billion
tokens, fixed micro-batch, gradient accumulation, and multi-GPU data parallelism, with the
optimizer, learning-rate schedule, and data pipeline frozen across variants. Quality is read off
held-out **validation cross-entropy / loss** on the same corpus (primary), language-model
**perplexity** on WikiText-2 and LAMBADA, and zero/few-shot **downstream accuracy** on ARC-Easy,
HellaSwag, PIQA, and WinoGrande. For the stability question, the natural diagnostic is the
magnitude of activations along the residual stream as a function of depth and training step, and
whether training survives at a given learning rate, depth, and precision rather than diverging.

## Code framework

The substrate is a standard GPT-style decoder in PyTorch: token and position embeddings feed a
stack of identical transformer blocks, then a final norm and a tied output projection to vocab
logits; the data pipeline, optimizer, schedule, and the attention and feed-forward modules
already exist and are not what is being designed. The two regions that are open are (a) the
normalization module and (b) how a block composes its sublayers, its normalization, and its
residual. They are left as stubs below; the attention and feed-forward modules are taken as
given.

```python
import torch
import torch.nn as nn


class Norm(nn.Module):
    """Per-block activation normalization with a learned per-channel gain.
    Keeps the (ndim, bias) signature so it is interchangeable in the block.
    The normalization rule itself is the open slot."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = 1e-5
        # TODO: any extra parameters/state the normalization rule we choose needs.

    def forward(self, x):
        # TODO: the normalization rule we will design — map x to a rescaled x
        #       using a statistic over the channel dimension and the learned gain.
        pass


class CausalSelfAttention(nn.Module):
    """Existing multi-head causal self-attention sublayer. Unchanged."""
    def __init__(self, config):
        super().__init__()
        # ... existing qkv projection, attention, output projection ...
    def forward(self, x):
        ...


class MLP(nn.Module):
    """Existing position-wise feed-forward sublayer. Unchanged."""
    def __init__(self, config):
        super().__init__()
        # ... existing up-projection, activation, down-projection ...
    def forward(self, x):
        ...


class Block(nn.Module):
    """One transformer block: normalization + the two sublayers + the residual.
    How the norm(s), attention, MLP, and the residual connection are wired
    together is the open slot."""

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        # TODO: the normalization module(s) and any wiring this block needs.

    def forward(self, x):
        # TODO: the block structure we will design — combine x, the normalization,
        #       attn, mlp, and the residual connection into the block's output.
        pass


# existing decoder stack and training loop the block plugs into
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # token embeddings
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # position embeddings
        self.h = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = Norm(config.n_embd, bias=config.bias)           # final norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))
        for block in self.h:
            x = block(x)                # the block whose internals we are designing
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits                   # (cross-entropy against targets handled by the loop)
```

The block's `forward` and the `Norm.forward` are the two empty slots; everything around them —
embeddings, the stack, the final norm, the head, and the training loop — is fixed.
