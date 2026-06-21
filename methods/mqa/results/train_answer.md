Attention-based sequence models train fast and generate slowly, and I want to understand exactly why and fix it. Training pushes a whole sequence through at once, every position in parallel, and the matmuls are big and dense — the hardware loves them. But autoregressive generation is strictly one token at a time, because the token I sample at position $t$ becomes the input at position $t+1$, and at every step the self-attention layer must look back over everything that came before: the query at the new position attends to the keys and values of all earlier positions. If I just count flops, incremental decoding does the same total arithmetic as the parallel forward, yet it is far slower. So flop-counting is lying about where the time goes. The right currency, on accelerators where arithmetic throughput is roughly two orders of magnitude higher than memory bandwidth, is the ratio of memory bytes moved to arithmetic performed: if each loaded byte is reused in many operations the layer is compute-bound and fast, and if that ratio creeps toward one the arithmetic units idle waiting for operands and the layer is memory-bandwidth bound.

Setting up the bookkeeping with the standard assumptions $m = n$, $k = v = d/h$, and $n \le d$, batched training attention does arithmetic $\Theta(b\,n\,d^2)$ while touching memory $O(b\,n\,d + b\,h\,n^2 + d^2)$, for a ratio $O(1/k + 1/(b\,n))$ — comfortably small, because each loaded key and value is dotted against all $n$ query positions and the fetch cost is amortized. Incremental decoding does the same total arithmetic $\Theta(b\,n\,d^2)$, but across the $n$ separate one-position calls it must reload the cached $K$ and $V$ every step, which summed over the sequence costs $\Theta(b\,n^2 d)$, plus $\Theta(n\,d^2)$ to reload the projection tensors. Dividing,

$$\frac{\text{memory}}{\text{arithmetic}} = \Theta\!\left(\frac{n}{d} + \frac{1}{b}\right),$$

which is near $1$ when $n \approx d$ or the batch is small. That is the diagnosis, derived rather than guessed: decoding stalls on bandwidth spent reloading the key/value cache. The $1/b$ term is trivial — use a bigger batch. The hard term is $n/d$, and it traces directly to the cached $K$ and $V$: at full length one of them stores $b\cdot h\cdot m\cdot k = b\,n\,d$ elements under $m=n$, $k=d/h$, and that $b\,n\,d$ is built as $h$ separate $d/h$-wide slices. The obvious ways to shrink it fall short. Cutting the number of heads $h$, or shrinking $d_k$ and $d_v$, does reduce the cache, but it degrades quality far out of proportion to the state saved, because the $h$ heads are the model's parallel read patterns — each head is an independent projection into a $d/h$-subspace, an independent question asked of the same memory, and the output projection mixes those reads. Throwing them away is throwing away the representational power I was trying to protect. The other family — local windows or other compression of the number of attended positions — attacks the $n$ factor instead and is genuinely useful, but it is orthogonal: within any window each head still keeps its own $K$ and $V$, so the head multiplicity in the cache is untouched.

I propose Multi-Query Attention. The break in the entanglement is to notice that the $h$-fold multiplicity matters asymmetrically. On the query side the $h$ heads are $h$ different questions, and together with the $h$-way output projection that mixes the per-question answers, that is where the multi-head expressiveness lives. On the key/value side the $h$ heads are $h$ projections of the same underlying memory — the answers written into that memory — and one shared set of answers, read by $h$ different questions, retains most of the power while $h$ separate sets are largely redundant. So I keep all $h$ query heads and the $h$-way output projection, and I remove the head dimension only from the keys and values: one shared key head and one shared value head, read by every query head. Many places that read the memory, one place that writes it. In the standard layer head $i$ computes $\text{logits}_i = q_i \cdot K_i^\top$, $\text{weights}_i = \mathrm{softmax}(\text{logits}_i)$, $o_i = \text{weights}_i \cdot V_i$, and $y = \sum_i o_i \cdot P_{o,i}$; sharing $K$ and $V$ simply drops the head index from them, so $\text{logits}_i = q_i \cdot K^\top$ against the single shared $K$, $o_i = \text{weights}_i \cdot V$ against the single shared $V$, and the output mixing $y = \sum_i o_i \cdot P_{o,i}$ is unchanged. Each query head still produces its own logits, its own attention pattern, and its own mixed output — the reads stay genuinely different — they just read a common $K$ and $V$. In the projection tensors $P_q$ stays $[h,d,k]$ and $P_o$ stays $[h,d,v]$, while $P_k$ drops from $[h,d,k]$ to $[d,k]$ and $P_v$ from $[h,d,v]$ to $[d,v]$. In einsum it is a one-index edit: wherever $K$, $V$, $P_k$, or $P_v$ carried an $h$, delete it.

The point is that this fixes the ratio I derived. The arithmetic is unchanged at $\Theta(b\,n\,d^2)$. With shared $K$ and $V$ the cached keys and values have no head dimension, so each is of size $b\cdot m\cdot k = b\,n\,(d/h)$ per length, and summed over the $n$ decode steps that term becomes $\Theta(b\,n^2 k) = \Theta(b\,n^2 d/h)$; the activations contribute $\Theta(b\,n\,d)$ and the projection reloads $\Theta(n\,d^2)$. Dividing by the arithmetic,

$$\frac{\text{memory}}{\text{arithmetic}} = \Theta\!\left(\frac{1}{d} + \frac{n}{d\,h} + \frac{1}{b}\right),$$

so the offending $n/d$ has become $n/(d\,h)$ — reduced by exactly the factor $h$ I was hunting for. The same conclusion follows from the storage side: the bytes cached per generated token are $2 \cdot n_{\text{layers}} \cdot n_{\text{kv\_heads}} \cdot d_{\text{head}} \cdot \text{bytes}$, and going from $n_{\text{kv\_heads}} = h$ to $n_{\text{kv\_heads}} = 1$ collapses the cached key matrix from $h \times d_{\text{head}}$ to $1 \times d_{\text{head}}$, shrinking both the per-token KV cache and the per-step reload bandwidth by $h$. With $h=8$ that is an $8\times$ cut in the dominant bandwidth term, turning decoding back toward compute-bound once a decent batch handles the $1/b$ piece.

Two details keep the comparison honest and the layer numerically sound. Sharing $K$ and $V$ deletes $(h-1)$ heads' worth of key and value projection parameters, so to compare at equal capacity I widen the feed-forward hidden width to put those parameters back — for example $4096 \to 5440$ on the translation model and $8192 \to 9088$ on the language model. That is a fair-comparison control, not part of the mechanism, which is purely the shared $K/V$. And the scaled dot-product attention's $1/\sqrt{d_k}$ logit scaling, which keeps the softmax out of saturated low-gradient regions when $d_k$ is large, is unchanged by sharing; I can fold it into $P_q$ or $P_k$ since scaling the query or key projection by $1/\sqrt{d_k}$ is equivalent to scaling the logits, or pass it explicitly to a fused kernel. In native einsum form the layer is the one-index edit applied to both the batched forward and the incremental decode step, the latter being the case the whole derivation was about, where the reloaded cache now carries no head dimension:

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

For a contemporary dense-tensor codebase the same idea is realized by fusing the Q/K/V projections into one linear and leaning on a fused scaled-dot-product-attention kernel. The fused projection produces a full-width query (all $h$ heads) and a single head of key and value; the query reshapes to $(b, h, n, d_{\text{head}})$ and the key and value to $(b, 1, n, d_{\text{head}})$; then, because the kernel wants matching head counts, the one shared $K/V$ head is broadcast up to $h$ heads before the call — a logical expansion of one physical head, with no extra cached state — so the thing cached and reloaded in decoding is still the single $(k, v)$ head and the bandwidth win is preserved:

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
