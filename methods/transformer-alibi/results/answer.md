# ALiBi — Attention with Linear Biases

## Problem

A decoder-only transformer LM's cost is dominated by the training input length `L` (attention
is `O(L^2)`). Trained at `L` with the usual added position embeddings, it fails to
**extrapolate**: scored at `L_valid > L` its perplexity improves for only a few dozen extra
tokens and then degrades, because the position signal past `L` is out of distribution. Goal:
train short and cheap, run long, and keep perplexity — with no extra runtime, memory, or
parameters over the cheapest existing position method, and ideally no learned parameters.

## Key idea

Use **no position embeddings at all**. After the query–key dot product (and after the usual
`sqrt(d_k)` scaling), add to the attention scores a *static, non-learned* bias that is
**linear in the query–key distance**, with a fixed per-head slope:

```
softmax( q_i K^T  +  m · [-(i-1), ..., -2, -1, 0] )
```

The nearest key (distance 0) gets penalty 0; a key `d` steps back gets `-m·d`. The slope
`m > 0` is fixed per head. This is a recency inductive bias: distant query–key pairs are
penalized, the penalty growing linearly with distance at a head-specific rate. Because the
penalty is the same scalar rule at every distance, evaluation past `L` does not introduce a new
absolute-position vector. Position enters only the scores (never the values) and at every layer,
so the residual stream carries no explicit absolute position.

## Slopes

Per head, slopes are a geometric sequence in `(0, 1)`, densest near 0 (the long-range heads).
For `n` a power of 2, start = ratio = `2^(-8/n)`, giving `2^(-8/n), 2^(-16/n), ..., 2^(-8)`:

- 8 heads: `1/2, 1/4, 1/8, ..., 1/256` (= `2^-1 ... 2^-8`).
- 16 heads: start `1/sqrt(2)`, ratio `1/sqrt(2)` (`2^-0.5, 2^-1, ..., 2^-8`), a half-step
  refinement of the 8-head grid that keeps the original integer powers and adds the intervening
  half-integer powers.

For non-power-of-2 `n`: take the closest-power-of-2 set, then append every-other slope from the
next power-of-2 set until `n` slopes are collected. Slopes are **fixed before training**: making
them trainable did not give strong extrapolation and slowed training by about 3%.
The bias is **not** scaled by `sqrt(d_k)`.

## Why it works / costs nothing

- **Defined everywhere as the same rule.** A linear penalty at distance 5000 is just `5000·m`:
  the same scalar distance rule continued beyond the training length, rather than a new absolute
  position vector. This directly targets the sinusoidal failure mode.
- **No extra ops.** The bias folds into the additive causal mask that is already applied before
  softmax — no operation is added, so there is no runtime penalty.
- **Translation-invariance trick.** softmax is invariant to a per-row constant and the bias is
  linear, so in zero-based columns the finite entry `-m_h(i-j)=m_hj-m_hi` can be replaced by the
  broadcast row `m_h[0,1,...,L-1]`, with the causal `-inf` mask removing future columns. The two
  rows differ only by `-m_hi`, so they produce the same attention weights.
- **Memory.** The mask grows from `L×L` to `n×L×L` (one slope per head), a small absolute
  increase (~0–0.7% net vs. sinusoidal at the same `L`); training short then recovers large
  `L^2` savings.
- **What the gains are.** Under stride-1 sliding-window inference (which removes the early token
  curse), ALiBi's perplexity stays **flat** as `L_valid` grows (sinusoidal explodes). So the
  nonoverlapping-inference gains from longer inputs come from reducing the early token curse,
  not (yet) from attending beyond `L_train` — but ALiBi matches/beats alternatives at
  `L_valid = L_train` while being simpler and parameter-free, and beats the cheap alternative at
  `L_valid > L_train`.

## Code (decoder-only, single attention sublayer)

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def alibi_slopes(n_heads):
    # Geometric slopes in (0,1), densest near 0. Power-of-2: start = ratio = 2^(-8/n).
    def pow2(n):
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3.0)))   # == 2^(-8/n)
        return [start ** (i + 1) for i in range(n)]
    if math.log2(n_heads).is_integer():
        return torch.tensor(pow2(n_heads))
    closest = 2 ** math.floor(math.log2(n_heads))
    extra = pow2(2 * closest)[0::2][: n_heads - closest]    # interleave from next pow-of-2 set
    return torch.tensor(pow2(closest) + extra)


class ALiBiAttention(nn.Module):
    """Causal multi-head attention with a linear per-head distance bias.
    No position embeddings; bias is added to scores before softmax."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.register_buffer("slopes", alibi_slopes(n_heads), persistent=False)  # [H]

    def _bias(self, T, device, dtype):
        pos = torch.arange(T, device=device, dtype=dtype)                         # [T]
        bias = self.slopes.to(device, dtype)[:, None, None] * pos[None, None, :]  # [H,1,T]
        causal = torch.triu(torch.full((T, T), float("-inf"), device=device, dtype=dtype), 1)
        return (bias + causal)[None, ...]                                         # [1,H,T,T]

    def forward(self, x):                       # x: [B, T, d_model], no position added
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)      # sqrt(d_k) scaling
        scores = scores + self._bias(T, x.device, scores.dtype)     # bias NOT scaled by sqrt(d_k)
        attn = F.softmax(scores, dim=-1)
        return self.out((attn @ v).transpose(1, 2).reshape(B, T, -1))
```
