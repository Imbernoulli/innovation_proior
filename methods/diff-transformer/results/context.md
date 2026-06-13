# Context: Attention noise in Transformer language models

## Research question

A decoder-only Transformer computes, at each position, a softmax-weighted average of value vectors:

```
o_m = sum_n a_{m,n} v_n,   a_{m,n} = softmax_n( q_m^T k_n / sqrt(d) )
```

The softmax is a probability distribution over *all* context positions, and because `exp(·) > 0`
everywhere, every position receives strictly positive weight. On a long context most positions are
irrelevant to the current prediction, yet the model is forced to spend a non-trivial fraction of the
attention mass on them. Empirically, trained Transformers allocate only a small share of their attention
to the few tokens that actually carry the answer and leak the rest across the irrelevant remainder — a
phenomenon visible as "attention noise" and as the *lost-in-the-middle* failure, where information placed
in the middle of a long context is under-retrieved.

The question: can the attention operator be changed so that it concentrates on the relevant context and
*cancels* the noise floor spread over the irrelevant tokens — while keeping the same parameter and FLOP
budget, the same causal-masking structure, and the same drop-in placement inside an otherwise standard
Transformer block?

## Background

**Scaled dot-product / multi-head attention (Vaswani et al., 2017).** The substrate. `h` heads each
project to `q, k, v` of dimension `d = d_model / h`, form `softmax(q k^T / sqrt(d)) v`, and concatenate.
The single softmax per head is the object every downstream method keeps; the noise floor is a property of
that single softmax.

**The positive-weight floor of softmax.** Because softmax outputs are strictly positive and sum to one,
attention can never assign exactly zero weight to an irrelevant token; it can only make the weight small.
With thousands of context positions, the aggregate of those small weights is a real fraction of the mass.
There is no mechanism inside a single softmax to *subtract* mass back off.

**Sparse / local attention (Child et al., 2019; Beltagy et al., 2020).** Restrict each query to a
hand-designed subset of keys (strided, local windows, global tokens). This removes mass from
out-of-window positions by construction, but the sparsity pattern is *fixed and structural*, not learned
from content — it cannot decide that a particular in-window token is noise, and it changes the
computational structure (which positions are even computed) rather than the score itself.

**Normalization and stability fixes (QK-Norm; logit soft-capping).** A separate line addresses the
*scale* of the logits — normalizing q/k or capping logits to keep the softmax in a responsive range.
These stabilize training but do not change the fundamental positive-weight floor: a normalized softmax is
still a strictly-positive distribution that cannot cancel its own noise.

**The analogy that frames the fix.** In electrical engineering a *differential amplifier* rejects the
voltage common to its two inputs and amplifies only their difference, cancelling common-mode noise;
noise-cancelling headphones subtract a second signal to remove the ambient floor. The shared principle:
the desired signal survives a subtraction of two correlated quantities, the common-mode noise does not.
If two attention maps over the same context share a common noise floor but differ in where they place
their *signal*, subtracting one from the other should cancel the floor and leave the signal — provided the
subtraction is calibrated so the result still behaves like an attention operator.

## Baselines

**Single-softmax multi-head attention (above).** Core idea: one positive softmax per head averages
values. Gap: the positive-weight floor means irrelevant context always draws mass; no way to cancel it.

**Fixed sparse / windowed attention.** Core idea: pre-decide which positions are visible. Gap: the
sparsity is structural and content-blind; it cannot learn that an in-context token is noise, and it alters
which positions are computed.

**Logit-scale / QK normalization.** Core idea: control the magnitude of the scores so the softmax stays
responsive. Gap: addresses scale, not the positive-weight floor; the distribution is still everywhere
positive.

## Evaluation settings

The natural yardsticks for a drop-in attention change at language-model scale:

- **Language-modeling perplexity** on a held-out validation set, across model sizes and training-token
  counts, against a parameter- and FLOP-matched vanilla Transformer (the scaling comparison).
- **Long-context modeling**: validation loss as the context length is extended, and key-information
  retrieval (needle-in-a-haystack / multi-needle) accuracy at long context.
- **Downstream**: in-context-learning accuracy and its robustness to example-order permutation;
  hallucination on summarization/QA; activation-outlier magnitude (relevant to quantization).

Tooling assumed: a standard decoder-only Transformer training stack (PyTorch), RoPE position encoding,
pre-LN blocks, RMSNorm, and a fused attention kernel.

## Code framework

A single multi-head attention block. The score-formation step is the slot to redesign; everything else —
the q/k/v projections, RoPE, causal mask, output projection — is standard.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F


def lambda_init_fn(depth):
    # depth is the 0-based layer index; the paper writes 0.8 - 0.6*exp(-0.3*(l-1)) for 1-based l.
    return 0.8 - 0.6 * (2.718281828 ** (-0.3 * depth))


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, depth):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, rel_pos, attn_mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k = apply_rotary(q, k, rel_pos)
        att = (q @ k.transpose(-1, -2)) * self.scaling
        if attn_mask is not None:
            att = att + attn_mask
        att = F.softmax(att, dim=-1)                  # the single-softmax slot to redesign
        o = att @ v
        o = o.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(o)
```

The slot to fill is the score-formation step: replace the single positive softmax with something that
concentrates on the relevant context and cancels the common-mode noise floor over the irrelevant
positions, at matched parameters and FLOPs.
