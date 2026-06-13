# Multi-Query Attention (MQA), distilled

Multi-Query Attention is a variant of multi-head attention in which the `h` attention heads keep
their own separate query projections (and the output projection), but **share a single key head and
a single value head**. Queries stay multi-headed; keys and values are reduced to one head reused
across all query heads. This shrinks the key/value tensors — and therefore the per-step
memory-bandwidth cost of autoregressive incremental decoding, and the size of the KV cache — by a
factor of `h`, with only minor quality loss relative to full multi-head attention.

## Problem it solves

Attention-based sequence models train fast (all positions in parallel) but generate slowly:
autoregressive decoding is one position at a time, and each step reloads the cached keys/values of
all prior positions. On accelerators where arithmetic throughput is ~100× memory bandwidth, this
makes incremental decoding **memory-bandwidth bound**, not compute bound.

## The diagnosis (why decoding is slow)

Count the ratio of memory bytes accessed to arithmetic operations, under `m = n`, `k = v = d/h`,
`n ≤ d`:

- **Batched (training) attention:** arithmetic `Θ(b n d²)`, memory `O(b n d + b h n² + d²)`, so the
  ratio is `O(1/k + 1/(b n))` — small; each loaded K/V is reused across all `n` queries.
- **Incremental (decode) attention:** arithmetic `Θ(b n d²)`, memory `Θ(b n² d + n d²)`, so the ratio
  is `Θ(n/d + 1/b)` — near 1 when `n ≈ d` or `b ≈ 1`, hence bandwidth-bound. The `1/b` term is fixed
  by larger batches; the offending `n/d` term comes from reloading the K/V cache each step. At full
  length, one of `K` or `V` has size `b·h·m·k = b·n·d` under `m=n`, `k=d/h`; summing those reloads over
  `n` decode steps gives the `Θ(b n² d)` term.

## Key idea

Remove the head dimension from the keys and values only. Keep `h` query heads and the `h`-way output
projection (this is where multi-head representational power lives — `h` parallel read patterns), but
project keys and values once and share them across all query heads.

In projection-tensor terms: `P_q : [h, d, k]` and `P_o : [h, d, v]` stay; `P_k` drops from `[h,d,k]`
to `[d, k]` and `P_v` drops from `[h,d,v]` to `[d, v]`. In einsum, delete the `h` index wherever it
indexes `K`, `V`, `P_k`, or `P_v`.

Re-deriving the incremental ratio with shared K/V: memory `Θ(b n d + b n² k + n d²)`, so the ratio
becomes `Θ(1/d + n/(d h) + 1/b)`. The `n/d` term is reduced by exactly `h`. Equivalently, the KV cache
per token is `2 · n_layers · n_kv_heads · d_head · bytes`; standard multi-head attention has
`n_kv_heads = h`, while shared-K/V attention has `n_kv_heads = 1`, so the cache and per-step reload
bandwidth shrink by `h`.

## Why share K/V rather than reduce `h` or `d_k`

Both fewer heads and smaller per-head dimension also shrink the K/V state, but they degrade quality
sharply — they remove the parallel read patterns / per-head capacity that make multi-head attention
strong. MQA's asymmetry (many query heads, one K/V head) keeps the read patterns and discards only the
redundancy of having `h` separate projections of the same memory, so quality stays close to baseline.
"One write-head is all you need."

## Design details

- **Parameter matching:** sharing K/V removes `(h−1)` heads of K/V projection parameters; widen the
  feed-forward hidden width (e.g. `4096 → 5440`, `8192 → 9088`) to restore the baseline's total
  parameter count for a fair, equal-capacity quality comparison. This is a control, not part of the
  mechanism.
- **Logit scaling:** the standard `1/√d_k` softmax scaling is unchanged; it can be folded into `P_q`
  or `P_k`, or applied explicitly by a fused attention kernel.
- **Orthogonality:** MQA reduces the head multiplicity of K/V; it composes with methods that limit or
  compress the number of attended positions `n` (e.g. local windows), which attack a different term.

## Final form (einsum)

```python
import tensorflow as tf  # einsum: named-index tensor contraction


def MultiqueryAttentionBatched(X, M, mask, P_q, P_k, P_v, P_o):
    """Multi-query attention. h query heads; ONE shared key head and value head.
    X: [b, n, d]   M: [b, m, d]   mask: [b, h, n, m]
    P_q: [h, d, k]   P_k: [d, k]   P_v: [d, v]   P_o: [h, d, v]  ->  Y: [b, n, d]"""
    Q = tf.einsum("bnd,hdk->bhnk", X, P_q)        # queries keep h heads
    K = tf.einsum("bmd,dk->bmk", M, P_k)          # shared key head (no h)
    V = tf.einsum("bmd,dv->bmv", M, P_v)          # shared value head (no h)
    logits = tf.einsum("bhnk,bmk->bhnm", Q, K)
    weights = tf.softmax(logits + mask)
    O = tf.einsum("bhnm,bmv->bhnv", weights, V)
    Y = tf.einsum("bhnv,hdv->bnd", O, P_o)        # h head-outputs projected and summed
    return Y


def MultiquerySelfAttentionIncremental(x, prev_K, prev_V, P_q, P_k, P_v, P_o):
    """One decode step; the reloaded caches prev_K/prev_V have NO head dimension."""
    q = tf.einsum("bd,hdk->bhk", x, P_q)
    K = tf.concat(
        [prev_K, tf.expand_dims(tf.einsum("bd,dk->bk", x, P_k), axis=1)], axis=1)
    V = tf.concat(
        [prev_V, tf.expand_dims(tf.einsum("bd,dv->bv", x, P_v), axis=1)], axis=1)
    logits = tf.einsum("bhk,bmk->bhm", q, K)
    weights = tf.softmax(logits)
    o = tf.einsum("bhm,bmv->bhv", weights, V)
    y = tf.einsum("bhv,hdv->bd", o, P_o)
    return y, K, V                                 # caches stay headless -> h-fold less reload
```

## Final form (fused projection + fused attention kernel)

A contemporary realization fuses the Q/K/V projections into one linear and uses a fused
scaled-dot-product-attention kernel. The single shared (K, V) head is broadcast up to `h` heads for
the kernel — a logical expansion only; just one head is ever cached.

```python
import math
import torch
import torch.nn as nn


def expand_kv_to_q_heads(t, target_heads):
    """Broadcast the shared KV head up to the query-head count for the kernel.
    Logical only: no extra (k, v) is cached, so decode reload stays h-fold smaller."""
    cur = t.size(1)
    if cur == target_heads:
        return t
    full = target_heads // cur
    rem = target_heads % cur
    parts = []
    if full > 0:
        parts.append(t.repeat_interleave(full, dim=1))
    if rem > 0:
        parts.append(t[:, :rem, :, :])
    return torch.cat(parts, dim=1)


class CausalSelfAttention(nn.Module):
    """Multi-query attention: h query heads, ONE shared (key, value) head."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        self.n_kv_head = 1                                   # one shared KV head
        kv_dim = 2 * self.n_kv_head * self.head_dim
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + kv_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)  # P_o
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        b, n, c = x.size()
        qkv = self.c_attn(x)
        q, kv = qkv.split([self.n_embd, 2 * self.n_kv_head * self.head_dim], dim=2)
        k, v = kv.chunk(2, dim=2)
        q = q.view(b, n, self.n_head, self.head_dim).transpose(1, 2)        # (b, h, n, d_head)
        k = k.view(b, n, self.n_kv_head, self.head_dim).transpose(1, 2)     # (b, 1, n, d_head)
        v = v.view(b, n, self.n_kv_head, self.head_dim).transpose(1, 2)     # (b, 1, n, d_head)
        k = expand_kv_to_q_heads(k, self.n_head)    # broadcast 1 -> h (logical)
        v = expand_kv_to_q_heads(v, self.n_head)
        y = torch.nn.functional.scaled_dot_product_attention(               # softmax(QKᵀ/√d_head)V
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0, is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, n, c)                    # concat heads
        y = self.resid_dropout(self.c_proj(y))                             # P_o
        return y
```
