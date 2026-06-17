# RoPE + QK-Norm, distilled

The combined attention recipe for GPT-style pretraining makes two orthogonal changes inside the
self-attention block. **RoPE (Rotary Position Embedding)** replaces learned absolute position
embeddings: it rotates each per-head query/key 2-plane by an angle proportional to the token's
position, so the attention logit depends only on the *relative* offset, with the sign determined by
which side carries the relative rotation. **QK-Norm**
normalizes the per-head queries and keys (RMSNorm along the head dimension) *before* the dot
product, bounding `||q||` and `||k||` so the attention logits cannot grow unbounded over
training. Because RoPE is a rotation (orthogonal, norm-preserving), the magnitude control that
QK-Norm installs survives the rotation, so the two stack with no interference and the logit ends
up both relative-only and magnitude-bounded.

## Problem it solves

Two persistent defects of the standard GPT-2 attention block, at once:

1. **Position is absolute, additive, length-capped.** Learned `wpe` (and additive sinusoidal)
   inject position by addition; expanding `(x_m + p_m)^T W_q^T W_k (x_n + p_n)` leaves cross
   terms depending on absolute `m, n`, not on the offset `m − n` language cares about, and the
   learned table caps the context length.
2. **Attention logits grow unbounded.** `q·k = ||q|| ||k|| cos` has no ceiling; the softmax
   sees only logit differences, so large magnitudes saturate it into near-one-hot, near-zero-
   entropy rows. `1/sqrt(d_k)` only calibrates the *expected* logit scale at initialization; as
   the projections move during training the norms grow and the logits climb, which at scale
   shows up as divergent training loss after a few thousand steps (attention-logit-growth /
   attention-entropy-collapse instability).

## Key idea

**RoPE.** Demand `<f_q(x_m, m), f_k(x_n, n)> = g(x_m, x_n, m − n)` with boundary `f(x, 0) = Wx`.
In 2D (`R^2 ≅ C`) the norm-preserving solution makes position a pure rotation: `f(x_m, m) = (W
x_m) e^{i m theta}`. Tile across `d/2` independent 2-planes at geometric frequencies `theta_i =
10000^{-2(i-1)/d}` for head dimension `d`; the block-diagonal rotation `R^d_{Theta,m}` is orthogonal, and since
rotations compose,

  q_m^T k_n = x_m^T W_q^T R^d_{Theta, n−m} W_k x_n,

so the logit depends only on the offset. The apparent `n−m` versus `m−n` sign is a convention:
with the relative matrix acting on the key side, `(R_m q)^T(R_n k)=q^T R_{n−m}k`; in the complex
product it is equivalently `Re[q k* e^{i(m−n)theta}]`. No learned position table, no length cap;
norm-preserving.

**QK-Norm.** The magnitudes are what `1/sqrt(d_k)` cannot bound. Fix them: pass the per-head `q`
and `k` through RMSNorm along the head dimension (`RMSNorm(a) = a / sqrt(mean(a_i^2) + eps)`,
no learned gain in this form). For nonzero vectors with negligible epsilon this gives
`||q||, ||k|| = sqrt(d_k)`; including epsilon and zero vectors, it gives
`||q||, ||k|| <= sqrt(d_k)`. Then `|q·k| <= d_k` by Cauchy-Schwarz, a fixed ceiling — the
bound `1/sqrt(d_k)` never gave. Normalize `q` and `k`
only (not `v`, which is averaged not scored), per head, along the head dimension (after the
multi-head split — that is the dimension the dot product contracts).

**Why keep `1/sqrt(d_k)` (not replace it with a learnable temperature).** This is the training-
stability variant of QK-norm, not the cosine-attention variant (Henry et al. 2020) that
L2-normalizes to unit length and swaps `1/sqrt(d_k)` for a learned `g ≈ log2(L^2 − L)`. RMSNorm
*fixes* the per-vector scale near `sqrt(d_k)` rather than crushing it to a unit cosine, so the raw
dot product keeps the usual typical scale while having a hard ceiling of `d_k`; after the
`1/sqrt(d_k)` divide the typical logits are back in the `O(1)` range and the ceiling is
`sqrt(d_k)`. No separate temperature is needed in this scaffold variant.

**Why they combine cleanly.** RoPE fixes *where* position lives in the logit; QK-Norm fixes *how
big* the logit can get — orthogonal failure modes. In the no-gain RMSNorm variant they commute
mathematically because RoPE is orthogonal and the same epsilon-adjusted RMS denominator is used;
RMSNorm bounds the norm, the rotation preserves it, so each guarantee holds in the presence of the other.
The reference scaffold normalizes
then rotates.

## Final algorithm (per head, per layer)

```
# inputs: q, k, v of shape [B, n_head, T, d_k]
q = rms_norm(q, over head dim d_k)        # QK-Norm: bound/pin ||q||
k = rms_norm(k, over head dim d_k)        # QK-Norm: bound/pin ||k||
q = rope(q)                               # rotate plane i by position * theta_i
k = rope(k)                               # theta_i = 10000^{-2(i-1)/d_k}
att = softmax( q @ k^T / sqrt(d_k) )      # 1/sqrt(d_k) KEPT, causal mask
y   = att @ v                             # values NOT normalized
```

Defaults: RoPE base `10000`; RMSNorm applied to `q, k` with no learned gain (`F.rms_norm(·,
(d_k,))`); no learned absolute position embedding (`use_pos_emb = False`).

## Working code

Filling the two open slots of the attention module — `use_pos_emb = False` (RoPE carries
position) and the per-head `q`/`k` transform (RMSNorm then RoPE), with `1/sqrt(d_k)` and the
causal softmax left in place:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size),
            )
        self.use_pos_emb = False  # RoPE replaces learned position embeddings
        # RoPE frequencies: theta_i = 10000^{-2i/head_dim} for zero-indexed i
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def _apply_rope(self, x, seq_len):
        # rotate each 2-plane by angle = position * theta_i (split-half layout)
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat([y1, y2], dim=-1).type_as(x)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # QK-Norm (RMSNorm along head dim) THEN RoPE; rotation preserves the bounded norm.
        q = self._apply_rope(F.rms_norm(q, (q.size(-1),)), T)
        k = self._apply_rope(F.rms_norm(k, (k.size(-1),)), T)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # 1/sqrt(d_k) kept
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
```

## Relation to prior methods

- **vs. learned/sinusoidal absolute PE:** RoPE makes the logit relative-only (`m − n`), removes
  the length cap, and injects position multiplicatively per-token (so it composes to `R_{n−m}`
  and is even compatible with linear/kernelized attention, which never forms the `N×N` matrix).
- **vs. relative-bias families (Shaw 2018, Transformer-XL, T5, DeBERTa):** those edit the
  additive expansion with learned tables/biases living inside the `N×N` logit matrix; RoPE
  solves the relative constraint in closed form with no extra parameters.
- **vs. cosine-attention QK-Norm (Henry et al. 2020):** that L2-normalizes `q, k` to unit
  length and replaces `1/sqrt(d_k)` with a learned temperature; the variant here uses no-gain RMSNorm
  (norm bounded near `sqrt(d_k)`, not crushed to a unit cosine) and *keeps* `1/sqrt(d_k)`. Same
  bounded-logit goal, lighter change, aimed at training stability at scale.
- **vs. `1/sqrt(d_k)` alone:** that is an init-time variance calibration, not a bound; QK-Norm
  converts it into a property that holds for the whole run by bounding the per-vector magnitude.
