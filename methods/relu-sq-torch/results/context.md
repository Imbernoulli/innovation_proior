# Context: the Transformer feed-forward nonlinearity (circa 2020-2021)

## Research question

The cost of training Transformer language models has grown to the point where it dominates the
research budget. A large fraction of a decoder block's parameters and FLOPs live in its
position-wise feed-forward network (FFN): two dense maps with a pointwise nonlinearity between
them, applied independently at every sequence position. The FFN's inner width is typically four
times the model width, so it is where most of the per-token compute is spent. The activation
sitting between the two dense maps is, uniquely, a *free* design knob — changing it costs no new
parameters and almost no extra FLOPs. The question is whether a different pointwise nonlinearity
in that slot can raise the model's **sample efficiency** — lower loss for the same number of
training tokens/steps — and thereby reduce the total compute needed to reach a target quality.

## Background

By this point the Transformer (Vaswani et al. 2017) is the dominant sequence model, and scaling it
up reliably lowers loss, following a power law between training compute and quality (Kaplan et al.
2020). The decoder block alternates multi-head self-attention with a position-wise FFN. In the
T5/decoder-only convention without biases the FFN is

```
FFN_ReLU(x, W1, W2) = max(x W1, 0) W2,
```

i.e. project up to the inner width `d_ff = 4 d_model`, apply a pointwise activation, project back
down. The activation is the only nonlinearity in the block besides the attention softmax.

**The activations on the table, and their shapes.** The original Transformer uses the rectified
linear unit `ReLU(x) = max(x, 0)`: a hard gate by sign, zero for negative inputs, identity for
positive ones. The de-facto replacement in BERT- and GPT-style models is the Gaussian Error Linear
Unit (Hendrycks & Gimpel 2016),

```
GELU(x) = x Φ(x) = x · ½ [1 + erf(x / √2)]  ≈  0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³))),
```

where `Φ` is the standard-normal CDF; it weights an input by its value rather than gating by its
sign, smoothing ReLU's kink and letting small negative inputs leak through. A third common choice
is Swish (Ramachandran et al. 2017), `Swish_β(x) = x σ(β x)`, a smooth self-gated unit. The fact
that matters about all three is their **asymptotics**: as `x → ∞`, each grows essentially
**linearly** in `x` (ReLU is exactly `x`; GELU and Swish approach `x` because `Φ(x), σ(βx) → 1`).
For large positive pre-activations, their derivatives all approach a constant near 1 — the slope
saturates.

**Multiplicative / gated alternatives.** A separate line replaces the additive activation with a
*multiplicative* one. Gated Linear Units (Dauphin et al. 2017) compute the element-wise product of
two linear projections of the input, one of them squashed:

```
GLU(X) = (X W + b) ⊗ σ(X V + c),
```

with `⊗` the Hadamard product. Their stated reason for liking it is the gradient: writing the two
branches as `p = XW + b` and `q = XV + c`, the local variation of `p ⊗ σ(q)` includes
`dp ⊗ σ(q)`, a route through the linear branch that is *not* downscaled by a derivative of the gate.
That gives a multiplicative path along which gradients propagate through depth without the
attenuation a plain saturating nonlinearity imposes. Building on this, a family of GLU variants for
the Transformer FFN (Shazeer 2020) swaps the squashing function and adds a third weight matrix:

```
FFN_GLU(x)     = (σ(x W) ⊗ x V) W2,
FFN_ReGLU(x)   = (max(x W, 0) ⊗ x V) W2,
FFN_GEGLU(x)   = (GELU(x W) ⊗ x V) W2,
FFN_SwiGLU(x)  = (Swish₁(x W) ⊗ x V) W2,
FFN_Bilinear(x)= (x W ⊗ x V) W2.
```

These have three weight matrices instead of two; to keep the parameter and FLOP counts equal to the
original FFN, the inner width `d_ff` is reduced by a factor of `2/3` (so `W`, `V` each map to
`(2/3)d_ff` units). On a T5 span-filling pretraining benchmark the GEGLU and SwiGLU variants were
the strongest members of this set, ahead of the plain ReLU/GELU/Swish FFNs.

**Higher-order / rectified-polynomial nonlinearities.** Rectified polynomials,
zero below a threshold and polynomial above it, have been studied as activation functions in the
associative-memory literature (Krotov & Hopfield 2016): increasing the power makes each
stored-pattern contribution *sharper*, raising the memory capacity (with a polynomial scaling in the
number of neurons). They are described there as deliberately sharper than the linear rectifier, and
noted as not commonly used in practice. Separately, multiplicative interactions — bilinear forms
`f(x, z) = zᵀ W x + …` — are understood (Jayakumar et al. 2020) to strictly enlarge the function
class a single layer can represent compared to a purely additive layer; gating, attention, and GLU
are all instances. The approximate GELU's own series
`0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))` carries a higher-order term, so even the "smooth"
mainstream activation is not purely linear in its formula.

## Baselines

**ReLU FFN (Vaswani et al. 2017).** `max(xW1, 0) W2`. Hard sign-gate, no parameters in the
activation, cheap forward and backward (the backward is a 0/1 mask). Core idea: keep the positive
part, kill the negative part.

**GELU FFN (Hendrycks & Gimpel 2016).** `GELU(xW1) W2`, `GELU(x) = xΦ(x)`. Smooths ReLU's corner
and lets a little negative signal through, which tends to train slightly better than ReLU and is the
standard choice in large LMs.

**Swish FFN (Ramachandran et al. 2017).** `Swish₁(xW1) W2`, `Swish_β(x) = xσ(βx)`. A smooth,
self-gated unit, non-monotonic near the origin.

**GLU-variant FFNs (Shazeer 2020).** Replace the activation-of-one-projection by a Hadamard product
of two projections, one squashed: e.g. `FFN_ReGLU(x) = (max(xW, 0) ⊗ xV) W2`. Core idea: a
multiplicative gate gives an un-attenuated gradient path (Dauphin et al. 2017) and a richer
(second-order) interaction between input directions; GEGLU/SwiGLU give the best LM perplexities in
this family. They introduce a *third* weight matrix `V`, and to remain parameter- and FLOP-matched
to the two-matrix FFN they cut the inner width `d_ff` by `2/3`.

**Transformer++ (the strengthened baseline).** A Transformer with RMSNorm (Zhang & Sennrich 2019),
Swish activation, and a SwiGLU multiplicative branch in the FFN (benchmarked in Narang et al. 2021).
Core idea: stack the individually-useful modern modifications.

## Evaluation settings

The natural yardsticks for an FFN-activation change:

- **Auto-regressive language modeling perplexity / log-perplexity** on held-out data, the primary
  metric. Datasets: the One Billion Word Benchmark (LM1B; Chelba et al. 2014) with sequence length
  64 at ~35M parameters as a fast comparison task; the C4 corpus (Raffel et al. 2020) and PG19 (Rae
  et al. 2020) at sequence length 512 and ~110M-537M parameters in the T5 codebase; and a
  GPT-3-style (Brown et al. 2020) decoder-only pretraining setup at sequence length 1024 and ~1.9B
  parameters, with downstream one-shot task accuracy as a secondary metric.
- **Compute-to-quality**, i.e. accelerator-hours (or training steps) needed to reach a fixed target
  perplexity, swept across model sizes to read off the compute-vs-quality power law.
- **Codebases / training protocol:** Tensor2Tensor, T5 (Mesh-TensorFlow), and Lingvo, each with its
  default Transformer hyperparameters and regularization disabled — Adafactor (Shazeer & Stern 2018),
  10K warmup steps at learning rate 0.01, then reciprocal-square-root decay — so that an activation
  change is tested with *no* extra tuning.
- For the GPU-kernel / nanoGPT setting: GPT-2 Medium
  (24 layers, 16 heads, 1024-wide, ~355M params) on FineWeb with the GPT-2 tokenizer; metrics are
  validation cross-entropy loss and training throughput (wall-clock). The FFN's matmul-activation-
  matmul core is isolated behind a single function so the activation can be changed in one place.

## Code framework

The existing nanoGPT-style FFN already exists in full: a two-matrix feed-forward block whose
up-projection `w_fc` maps to the inner width `4·n_embd`, whose down-projection `w_proj` maps back,
and whose pointwise nonlinearity between them is the one slot to be designed. The MLP module owns the
two linear weights and the dropout; the matmul-activation-matmul core is factored into a single
`fused_mlp_forward(x, w_fc, w_proj)` so the activation lives in exactly one place. The default core
uses a stock pointwise activation; what goes between the two matmuls — and how its backward is
computed — is the empty slot.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F


def fused_mlp_forward(x, w_fc, w_proj):
    """FFN core: up-projection -> pointwise activation -> down-projection.

    x:      (B*T, n_embd)        input rows
    w_fc:   (4*n_embd, n_embd)   up-projection weight
    w_proj: (n_embd, 4*n_embd)   down-projection weight
    returns (B*T, n_embd)
    """
    h = x @ w_fc.t()                 # up-project to the 4x inner width
    # TODO: the pointwise nonlinearity to apply between the two matmuls,
    #       and (if we hand-write it) how its backward pass is computed.
    a = h                            # <- placeholder for the activation we will choose
    return a @ w_proj.t()            # down-project back to n_embd


class MLP(nn.Module):
    """Position-wise feed-forward block: the standard two-matrix FFN.
    Owns the up/down projections and dropout; the activation is delegated
    to fused_mlp_forward."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        out = fused_mlp_forward(x.view(-1, C), self.c_fc.weight, self.c_proj.weight)
        out = self.dropout(out.view(B, T, C))
        return out
```

The single empty slot is the activation between the two matmuls (and, if performance matters, the
hand-written backward that goes with it). Everything else — the two linear maps, the `4×` inner
width, dropout, the surrounding block and training loop — already exists.
