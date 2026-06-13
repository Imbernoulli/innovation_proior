# Multi-Head Attention, distilled

Multi-Head Attention (MHA) is a cross-positional layer that updates each position's
representation as a content-weighted sum over all positions: one dense matmul forms all
query-key scores, a softmax normalizes them, and a second matmul reads out values, run in parallel
across `h` heads. It replaces recurrence for sequence mixing:
every position reaches every other in one hop (`O(1)` path length) with `O(1)` sequential
operations, so it is fully parallelizable across positions.

## Problem it solves

Mix information across positions in a sequence so that any two positions — however far apart —
can influence each other, while (1) keeping the number of *sequential* operations independent of
sequence length, (2) keeping the maximum path length between positions short and constant, (3)
being fully parallelizable, and (4) costing no more per layer than the recurrent/convolutional
layers it replaces. Recurrence pays `O(n)` sequential ops and `O(|i-j|)` path length;
convolution parallelizes but its path length still grows with distance.

## Key idea

Treat the layer as a soft dictionary lookup. Each position emits a **query** (what it is looking
for), a **key** (what it advertises), and a **value** (what it contributes). The output for a
position is the softmax-weighted sum of all values, weighted by the compatibility of its query
with each key. Pack queries/keys/values into matrices `Q, K, V` (one row per position):

```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
```

- `Q K^T` is the `n×n` matrix of all pairwise scores in one matmul (dot-product / multiplicative
  score, chosen over an additive MLP score because it collapses to a single optimized dense GEMM
  — decisive when attention is used in every layer).
- Per-layer cost `O(n^2 · d)`, sequential ops `O(1)`, maximum path length `O(1)`.

**Scaling by `1/sqrt(d_k)`** — derived, not tuned. If the components of `q` and `k` are
independent with mean 0 and variance 1, then `q·k = Σ_{m=1}^{d_k} q_m k_m` has mean 0 and

```
Var(q·k) = E[(Σ_m q_m k_m)(Σ_l q_l k_l)]
         = Σ_m Σ_l E[q_m k_m q_l k_l]
         = Σ_m E[q_m^2]E[k_m^2] = d_k   (m≠l terms vanish by independence and zero mean)
```

so the logits have standard deviation `sqrt(d_k)`. Large-spread logits push the softmax into a
saturated region where, for `p = softmax(s)`, the Jacobian entry
`∂p_i/∂s_j = p_i(δ_ij − p_j)` is close to zero, so gradients through the weights become very
small. Dividing by `sqrt(d_k)` restores logit variance to 1 under
the unit-variance assumption. (This matches the observation that raw dot-product attention
degrades relative to additive attention as `d_k` grows; the additive score is not a bare sum of
`d_k` products and includes a bounded nonlinearity.)

**Multiple heads.** A single softmax-weighted sum is a convex combination — one weighted average
— so it blends distinct relations together and cannot attend to several different things at once.
Run `h` attention functions in parallel, each through its own learned low-dimensional projections,
keep the outputs separate, concatenate, and mix with a final projection `W^O`:

```
head_i      = Attention(Q W_i^Q, K W_i^K, V W_i^V)
MultiHead   = Concat(head_1, ..., head_h) W^O
W_i^Q, W_i^K ∈ R^{d_model × d_k},  W_i^V ∈ R^{d_model × d_v},  W^O ∈ R^{h d_v × d_model}
```

Each head can match in a different subspace (`W^Q, W^K`) decoupled from the content it carries
(`W^V`); `W^O` linearly recombines the heads. Setting `d_k = d_v = d_model / h` makes `h` heads
keep the same pairwise-attention order as one full-width head:
`h · O(n^2 · d_model/h) = O(n^2 · d_model)`. The Q/K/V/O projections still cost model-width
linear work, `O(n · d_model^2)`, as in a full-width attention layer; the head split prevents the
`n×n` score-and-value work from multiplying by `h`.

**Defaults and why:** `d_model = 512`, `h = 8`, `d_k = d_v = 64`. More heads track more distinct
relations but each gets a thinner subspace; `8` heads of width `64` balance the two within the
single-head compute budget.

**Causal masking.** For autoregressive decoding, position `i` must not see `j > i`. Set those
scores to `-inf` before the softmax; `exp(-inf) = 0`, so future positions get exactly zero
weight and the remaining causal weights renormalize to 1.

## Final algorithm

```
# x: (batch, seq_len, d_model)
Q, K, V  =  x W^Q,  x W^K,  x W^V                 # per head; in practice one fused projection
for each head i in 1..h:
    S_i      = Q_i K_i^T / sqrt(d_k)              # (n x n) scores
    S_i      = mask_future(S_i)                   # causal: set j>i entries to -inf (optional)
    A_i      = softmax(S_i, axis=-1)              # row-wise distribution over positions
    head_i   = A_i V_i                            # weighted sum of values
out  =  Concat(head_1, ..., head_h) W^O           # mix heads back to d_model
```

## Working code

Filling the `SequenceMixing` slot of the harness: packed QKV projection, reshape into heads,
scaled dot-product attention with optional causal mask, concat, output projection. The
fused-kernel path (`F.scaled_dot_product_attention`) computes the same
`softmax(QK^T/sqrt(d_k))V` operation, with causal masking when `is_causal=True`; the manual path
spells it out.

```python
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """softmax(Q K^T / sqrt(d_k)) V.  q,k,v: (batch, n_head, seq_len, d_k)."""
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)        # variance of logits back to 1
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))  # future logits -> -inf -> weight 0
    attn = scores.softmax(dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return attn @ v                                            # weighted sum of values


class SequenceMixing(nn.Module):
    """The cross-positional sub-layer implemented as h parallel scaled-dot-product
    heads, concatenated and projected back to d_model by W^O."""

    def __init__(self, config, causal=True):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_k = config.n_embd // config.n_head              # d_k = d_v = d_model / h
        self.causal = causal
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)  # fused Q,K,V
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)      # W^O
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)     # project, split into Q, K, V
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)  # (B, n_head, T, d_k)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )
        else:
            m = self.mask[:, :, :T, :T] if self.causal else None
            y = scaled_dot_product_attention(q, k, v, mask=m, dropout=self.attn_dropout)
        y = y.transpose(1, 2).contiguous().view(B, T, C)       # concat heads -> d_model
        y = self.resid_dropout(self.c_proj(y))                 # W^O
        return y
```

## Relation to prior methods

- **Bahdanau additive attention**: same weighted-sum structure `c_i = Σ_j α_ij v_j`, but the
  score is a one-hidden-layer MLP `v_a^T tanh(W_a q + U_a k)`, and it rides on an RNN. MHA
  replaces the MLP score with the scaled dot product (one GEMM) and uses it as the *only*
  cross-positional mechanism, dropping recurrence.
- **Luong multiplicative attention**: provides the `dot` score `q^T k` that MHA scales by
  `1/sqrt(d_k)`; still attached to an RNN and unscaled.
- **Self-attention vs recurrence/convolution**: `O(1)` sequential ops and `O(1)` path length vs
  recurrence's `O(n)/O(n)` and convolution's `O(1)/O(n/k or log_k n)`.
