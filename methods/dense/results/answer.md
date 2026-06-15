# Scaled Dot-Product Attention, distilled

Scaled dot-product attention is a content-addressed sequence-mixing layer. Each position emits
a query, a key, and a value; the output for a query is a softmax-weighted blend of all values,
weighted by the dot-product compatibility of that query with each key, with the scores divided
by `sqrt(d_k)` to keep the softmax in its high-gradient regime:

```
Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_k) ) V
```

`Q` is `(N × d_k)` (one query row per reader position), `K` is `(N × d_k)`, `V` is `(N × d_v)`.
All pairwise scores are one matmul `Q Kᵀ`; the softmax normalizes each query's row over the
keys; the blend is a second matmul against `V`. Per-layer cost `O(N² d)`, sequential ops
`O(1)`, maximum path length between any two positions `O(1)` — every position reaches every
other in a single hop. It is the "dense" full-attention kernel: every legal (query, key) pair
receives nonzero weight.

## Problem it solves

Mix information across all positions of a sequence as a single content-addressed read — any
position can pull from any other in one hop — that is (1) cheap enough to be the primary mixing
operation in every layer of a deep stack (one optimized matrix multiply, not a per-pair MLP),
(2) a proper normalized blend (nonnegative weights summing to one, smoothly differentiable),
(3) numerically stable as the model gets wide (the softmax must not saturate as `d_k` grows),
and (4) able to enforce causal masking (a position may not read the future) inside the same
operation. Recurrence pays `O(N)` sequential ops and `O(N)` path length; convolution
parallelizes but its path length grows with distance.

## Key idea

Treat the layer as a soft dictionary lookup. The compatibility of query `q` with key `k` is
the dot product `q · k` — chosen over an additive MLP score (Bahdanau) because all `N²` scores
collapse into one dense GEMM `Q Kᵀ`, decisive when attention runs in every layer. A softmax
over keys turns each query's score row into a convex combination of values.

**Scaling by `1/sqrt(d_k)` — derived, not tuned.** If the components of `q` and `k` are
independent with mean 0 and variance 1, then `q · k = Σ_{i=1}^{d_k} q_i k_i` has

```
E[q·k]   = Σ_i E[q_i] E[k_i] = 0
Var(q·k) = E[(q·k)²] = Σ_i Σ_j E[q_i k_i q_j k_j]
         = Σ_i E[q_i²] E[k_i²]   (i=j terms; i≠j terms vanish by independence + zero mean)
         = Σ_{i=1}^{d_k} 1 = d_k.
```

So the logits have standard deviation `sqrt(d_k)`: as the model gets wider the dot products
grow, push the softmax toward a near-one-hot distribution, and the softmax gradient
`∂softmax_i/∂z_j = softmax_i(δ_ij − softmax_j)` collapses toward zero, so learning stalls.
Dividing by `sqrt(d_k)` restores the per-logit standard deviation to 1 independent of width
(`Var = d_k / (sqrt(d_k))² = 1`), keeping the softmax in its high-gradient region. The factor
is the square root of the variance, not the variance: dividing by `d_k` would crush the logit
scale to `1/d_k` and over-smooth the weights. This variance argument predicts a failure mode
consistent with the observed large-width degradation of unscaled multiplicative attention; the
additive score does not sum `d_k` raw products before the softmax.

## Causal masking

For autoregressive decoding, query `i` may read only keys `j ≤ i`. Add a bias `M` to the
scores before the softmax with `M_ij = 0` for `j ≤ i` and `M_ij = −∞` otherwise; `exp(−∞) = 0`
removes the illegal entries from both numerator and denominator, so the softmax renormalizes
over exactly the legal keys:

```
Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_k) + M ) V.
```

Masking after the softmax would leave the denominator counting illegal keys and break the
sum-to-one normalization. With no mask (encoder self-attention), every position reads every
position — the dense, fully connected read over all `N²` pairs.

## Why these choices

- **Dot-product score over additive MLP** — all scores collapse to one optimized GEMM; the
  layer runs in every layer of the stack, so its constant factor dominates. Theoretical
  complexity is similar; hardware utilization is not.
- **Divide by `sqrt(d_k)`** — normalize the logit standard deviation to 1 so the softmax keeps
  large gradients; derived from `Var(q·k) = d_k`.
- **Softmax weights** — convex combination, output in the convex hull of values, smooth,
  differentiable, probabilistic reading.
- **Additive `−∞` mask before softmax** — renormalizes over the legal keys exactly.
- **PyTorch SDPA kernel** — `softmax(QKᵀ·scale + mask)V` as a single library primitive that can
  dispatch to fused backends; identical math, less memory traffic when a fused backend is used.
  The dense full-attention module routes through it for speed while keeping the explicit
  three-step form as the reference definition.

## Working code

Reference (explicit) form — the canonical scaled dot-product attention, structurally identical
to the standard Annotated Transformer implementation:

```python
import math
import torch


def attention(query, key, value, mask=None, dropout=None):
    """Scaled dot-product attention: softmax(Q Kᵀ / √d_k + mask) V."""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)   # Q Kᵀ / √d_k
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))              # illegal pairs → weight 0
    p_attn = scores.softmax(dim=-1)                                        # rows sum to 1 over legal keys
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn                            # blend values
```

Fused form — `torch.nn.functional.scaled_dot_product_attention`, which implements the same
operation and can build the causal `−∞` bias from `is_causal`. This is the operation the dense
full-attention module runs:

```python
import math
import torch.nn as nn
import torch.nn.functional as F


class DenseAttention(nn.Module):
    """Full (dense) scaled dot-product attention. Every legal (query, key) pair attends."""

    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p

    def forward(self, q, k, v, attn_mask=None, is_causal=False, scale=None, enable_gqa=False):
        # q, k, v: (B, H, N, D). scale defaults to 1/√D — the derived 1/√d_k.
        D = q.size(-1)
        scale = scale if scale is not None else 1.0 / math.sqrt(D)
        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
        )
        return out
```

The reference PyTorch pseudocode the fused kernel computes:

```python
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                 is_causal=False, scale=None, enable_gqa=False):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:                                   # lower-triangular −∞ bias
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    if attn_mask is not None:                       # bool → −∞ on disallowed; float → additive
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor             # Q Kᵀ / √d_k
    attn_weight += attn_bias                                               # mask before softmax
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value                                            # blend values
```
