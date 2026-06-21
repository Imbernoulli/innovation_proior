# Context: Attention in Transformer language models

## Research question

A decoder-only Transformer computes, at each position, a softmax-weighted average of value vectors:

```
o_m = sum_n a_{m,n} v_n,   a_{m,n} = softmax_n( q_m^T k_n / sqrt(d) )
```

The softmax is a probability distribution over *all* context positions, and because `exp(·) > 0`
everywhere, every position receives strictly positive weight. Empirically, trained Transformers allocate
only a small share of their attention to the few tokens that carry the answer and spread the rest across
the remaining context — visible as "attention noise" and as the *lost-in-the-middle* behavior, where
information placed in the middle of a long context is under-retrieved.

The question: how should the score-formation step inside an attention head be designed so that the head
concentrates on the relevant context — while keeping the same parameter and FLOP budget, the same
causal-masking structure, and the same drop-in placement inside an otherwise standard Transformer block?

## Background

**Scaled dot-product / multi-head attention (Vaswani et al., 2017).** The substrate. `h` heads each
project to `q, k, v` of dimension `d = d_model / h`, form `softmax(q k^T / sqrt(d)) v`, and concatenate.
Each head computes a single softmax over the context.

**The positive-weight property of softmax.** Softmax outputs are strictly positive and sum to one, so
every context position receives some weight; with thousands of positions the aggregate of the small
weights is a real fraction of the total mass.

**Sparse / local attention (Child et al., 2019; Beltagy et al., 2020).** Restrict each query to a
hand-designed subset of keys (strided, local windows, global tokens). Out-of-window positions are removed
from the computation by construction; the sparsity pattern is fixed and structural rather than learned
from content.

**Normalization and stability fixes (QK-Norm; logit soft-capping).** A separate line addresses the
*scale* of the logits — normalizing q/k or capping logits to keep the softmax in a responsive range —
which stabilizes training.

## Baselines

**Single-softmax multi-head attention (above).** Core idea: one positive softmax per head averages
values over the context.

**Fixed sparse / windowed attention.** Core idea: pre-decide which positions are visible to each query
via a structural pattern.

**Logit-scale / QK normalization.** Core idea: control the magnitude of the scores so the softmax stays
responsive.

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

The slot to fill is the score-formation step: redesign how the single positive softmax forms the
attention weights, at matched parameters and FLOPs.
