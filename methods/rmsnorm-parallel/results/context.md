# Context: the transformer block under scaling pressure

## Research question

A GPT-style decoder is a stack of identical transformer blocks; the same block runs once per
layer, and the largest models being trained stack dozens to over a hundred of them. Inside each
block there are two computational regions whose form is not fixed by anything fundamental: the
**normalization rule** (how activations are rescaled before each sublayer) and the **block
structure** (how the attention sublayer, the feed-forward sublayer, and the residual connection
are wired together). The default is LayerNorm with a learned scale and bias, placed in a
pre-normalization arrangement: `x + Attn(LN(x))`, then `x + MLP(LN(x))`. At the current scale,
the block is executed billions of times over a training run and is the unit replicated across
accelerators, so any per-block cost — an extra reduction, an extra synchronization point, a
sequential dependency between sublayers — is multiplied by the layer count and by the device
count and shows up directly in wall-clock training cost.

The question is how to redesign the normalization and wiring of the block — without touching
the dataset, tokenizer, optimizer, schedule, or the attention and feed-forward computations
themselves — in a way that reduces per-block cost under large-scale, multi-device training.

## Background

**Why normalization is in the block at all.** Normalization layers stabilize and speed up
training (measured in steps) by keeping the scale of activations and gradients under control as
signals pass through a deep stack. LayerNorm (Ba, Kiros & Hinton 2016) computes, for the summed
inputs `a ∈ R^n` to a layer, a mean and a standard deviation across the `n` channels and
standardizes: `ā_i = (a_i − μ)/σ · g_i (+ b_i)`, with `μ = (1/n)Σ a_i`, `σ = sqrt((1/n)Σ(a_i −
μ)^2)`, a learned per-channel gain `g` (init 1) and optional bias `b`. Unlike batch
normalization it uses no cross-example statistics, so it handles variable-length sequences and
is the de-facto choice in transformers. A standard analysis attributes its benefit to two
invariances it confers: **re-centering** invariance (the output is unchanged if inputs or
weights are shifted by a constant) from the mean-subtraction, and **re-scaling** invariance (the
output is unchanged if inputs or weights are scaled) from the division by `σ`. Santurkar et al.
(2018) argue the gain is less about input stability per se and more about smoothing the
optimization landscape. Each LayerNorm call does **two reductions over the channel dimension**
(one for the mean, one for the variance) plus an elementwise subtraction.

**Pre-LN vs Post-LN — where the norm sits.** The original transformer placed normalization
*after* the residual addition (Post-LN). Xiong et al. (2020) showed that in Post-LN the expected
gradients of the parameters near the output layer are large at initialization, which forces a
learning-rate warmup stage and makes training delicate; placing the norm *inside* the residual
branch (Pre-LN), so the residual stream itself is never normalized, gives well-behaved gradients
at initialization and trains stably without warmup. Pre-LN is therefore the prevailing choice
for large language models. The pre-normalized block has the shape `x ← x + Sublayer(LN(x))`: the
residual path is an identity highway, and each sublayer reads a freshly normalized copy of it.

**The cost of a block at scale, and where it is paid.** A model too large for one accelerator is
sharded: the matrices inside attention and the feed-forward layer are split across devices
(tensor / op-sharding, Shoeybi et al. 2019). Under op-sharding, every place where a sublayer's
sharded output must be summed back into the full-width residual stream requires an **all-reduce**
collective across the devices — one in the forward pass and one in the backward pass per such
point. Collectives are a synchronization barrier and a communication cost, and at large device
counts they are a leading term in step time. A pre-normalized block as written has two sublayers
added back to the residual in sequence, hence two such reduction points per block; it also has
two normalization layers, each its own elementwise op over the full width. Multiplied by the
number of layers and the number of devices, these per-block counts are exactly the levers that
move training throughput.

**The block's sequential structure.** In the standard block the feed-forward sublayer takes as
its input the residual stream *after* the attention sublayer has already been added to it:
`MLP(LN(x + Attn(LN(x))))`. The two sublayers are thus strictly ordered — the MLP's input
depends on attention's output. This ordering is inherited from the original design rather than
derived from a requirement; both sublayers are, by construction, perturbations added to the same
residual highway.

**Weight normalization as a precedent.** The Euclidean norm (without the `1/n` factor) had been
used for weight normalization (Salimans & Kingma 2016) to decouple the length of weight vectors
from their direction, offering re-scaling invariance for weight vectors during optimization.

## Baselines

**Standard pre-normalized LayerNorm block (Vaswani et al. 2017; Pre-LN per Xiong et al. 2020).**
The default. Two LayerNorms and two residual additions per block, in series:

```
x ← x + Attn(LN1(x))
x ← x + MLP(LN2(x))
```

Each `LN` does two channel-wise reductions (mean and variance) plus a subtraction; the residual
adds are sequential, so the MLP cannot begin until attention's contribution is in the stream.

**RMSNorm (Zhang & Sennrich 2019).** Drops the mean entirely and normalizes by the root-mean-square alone:

```
ā_i = a_i / RMS(a) · g_i,   RMS(a) = sqrt( (1/n) Σ_{i=1}^n a_i^2 )
```

with learned gain `g` (init 1) and no bias by default — there is no mean removed, hence no
re-centering invariance that a bias would need to restore. Because RMS is homogeneous under
scalar rescaling, `RMS(αa) = |α|·RMS(a)` (and `α·RMS(a)` for positive `α`), RMSNorm is
invariant to re-scaling of the inputs and of the weight matrix (the positive scale cancels) but
is **not** invariant to shifts and **not** to per-weight-vector re-scaling.
It does one reduction (sum of squares) and no subtraction, versus LayerNorm's two reductions and
a subtraction. Its gradient with respect to the weight matrix carries a factor inversely
correlated with weight scale, acting as an implicit per-layer learning-rate adaptor.

**Multi-query attention / SwiGLU / rotary embeddings (Shazeer 2019; Shazeer 2020; Su et al.
2021).** Other knobs in the block's design space at the time — sharing key/value heads for
cheaper decoding, a gated feed-forward activation, and a relative position scheme inside
attention. They modify the attention or the feed-forward internals while leaving the block-level
normalization and residual wiring arrangement unchanged.

## Evaluation settings

The natural yardstick is GPT-style decoder-only pretraining held fixed except for the block's
normalization and wiring. A representative protocol: a GPT-2-Medium-class model (24 layers, 16
heads, model width 1024, ≈355M parameters), pre-normalized decoder, trained on a large
web-text corpus with the GPT-2 byte-pair tokenizer over a few billion tokens, fixed micro-batch,
gradient accumulation, and multi-GPU data parallelism, with the optimizer, learning-rate
schedule, and data pipeline frozen across variants. Quality is read
off held-out **validation cross-entropy / loss** on the same corpus (primary), language-model
**perplexity** on WikiText-2 and LAMBADA, and zero/few-shot **downstream accuracy** on
ARC-Easy, HellaSwag, PIQA, and WinoGrande. Efficiency is read off training **throughput**
(tokens/second or step time) for the per-block cost the wiring changes are meant to reduce.

## Code framework

The substrate is a standard GPT-style decoder in PyTorch: token and position embeddings feed a
stack of identical transformer blocks, then a final norm and a tied output projection to vocab
logits; the data pipeline, optimizer, schedule, and the attention and feed-forward modules
already exist and are not what is being designed. The two regions that are open are (a) the
normalization module and (b) how a block composes its sublayers and residual. They are left as
stubs below; the attention and feed-forward modules are taken as given.

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
