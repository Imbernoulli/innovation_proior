# Grouped-Query Attention (GQA)

## Problem

Autoregressive Transformer decoding is bottlenecked by memory bandwidth, not
arithmetic: every decode step must stream the model weights and the entire
key-value (KV) cache from memory while doing little compute per token. For a
self-attention layer (batch $b$, length $n$, model dim $d$, $h$ heads, head dim
$k=d/h$), the per-sequence arithmetic is $\Theta(b\,n\,d^2)$ and the memory
access is $\Theta(b\,n^2 d + n\,d^2)$, giving a memory-to-compute ratio of

$$\frac{n}{d} + \frac{1}{b}.$$

The $n/d$ term comes from streaming the KV cache, whose size is $\propto h$ (one
key and one value head per position per head). When $n \approx d$ the layer is
memory-bound and decoding stalls.

Multi-query attention (MQA) shrinks the cache by sharing a *single* key/value
head across all $h$ query heads, cutting the offending term to $n/(dh)$ — fast,
but a single shared key/value subspace degrades quality and destabilizes
training. And existing multi-head checkpoints can't use it without retraining.

## Key idea

Make the number of key/value heads a dial. Partition the $H$ query heads into
$G$ **groups**; each group shares one key head and one value head. The KV cache
is then $\propto G$, and the memory ratio term is $nG/(dh)$:

- $G = 1$ recovers MQA (one shared KV head);
- $G = H$ recovers multi-head attention (MHA);
- intermediate $G$ interpolates — capacity spread over $G$ subspaces instead of
  one, cache cut by a factor $H/G$.

Because the KV cache scales with $d$ while compute scales with $d^2$, large
models are less cache-dominated, so the speed cost of raising $G$ is small near
$G=1$ and rises only as $G \to H$. A modest $G$ (e.g. 8) recovers near-MHA
quality at near-MQA speed, and aligns with tensor-parallel sharding (no replicated
single head). GQA is applied to decoder self-attention and cross-attention, not
encoder self-attention (the encoder runs in parallel and is not bandwidth-bound).

## Uptraining (cheap conversion from an existing checkpoint)

Don't train from scratch — convert a trained MHA checkpoint in two steps:

1. **Convert.** For each group, build the shared key/value projection by
   **mean-pooling** the projection matrices of the $H/G$ heads in that group:
   $$W_k^{g} = \tfrac{1}{|g|}\sum_{j\in g} W_k^{(j)}, \qquad
     W_v^{g} = \tfrac{1}{|g|}\sum_{j\in g} W_v^{(j)}.$$
   Mean-pooling preserves the group's average response and beats selecting one
   head or random initialization (ordered by information preserved). Query and
   output projections copy over unchanged.
2. **Adapt.** Continue pre-training on the original recipe/data for a small
   fraction $\alpha$ of the original steps ($\alpha \approx 0.05$; ~10% gives
   little extra). GQA is already reasonable right after conversion; MQA relies
   more heavily on this step.

## Algorithm

```
Given trained MHA checkpoint with H heads, target G groups (G | H):
  q_proj, o_proj          <- copy from checkpoint
  for each group g of H/G heads:
      W_k[g] <- mean over heads in g of trained W_k
      W_v[g] <- mean over heads in g of trained W_v
  continue pre-training alpha * original_steps on the original recipe

Inference (per decode step):
  q = q_proj(x)                      # H query heads
  k, v = k_proj(x), v_proj(x)        # G kv heads -> cache holds only G heads
  k, v = cache.append(k, v)
  k, v = repeat_kv(k, H/G), repeat_kv(v, H/G)   # expand to H, no extra cache
  out = o_proj( softmax(q kᵀ / sqrt(k)) v )
```

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[b, G, s, k] -> [b, G*n_rep, s, k]: repeat each of the G kv heads
    n_rep = H/G times so it serves its group of query heads. Only G heads are
    stored in the cache; the expansion happens after loading, pre-matmul."""
    b, G, s, k = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(b, G, n_rep, s, k)
    return x.reshape(b, G * n_rep, s, k)


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads                       # H query heads
        self.num_kv_heads = num_kv_heads                 # G groups
        self.num_kv_groups = num_heads // num_kv_heads   # group size = H/G = n_rep
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, x, kv_cache=None, mask=None):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.append(k, v)     # cache stores only G kv heads

        k = repeat_kv(k, self.num_kv_groups) # expand to H heads, no extra cache
        v = repeat_kv(v, self.num_kv_groups)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(out)


def convert_mha_to_gqa(mha, num_kv_heads):
    """Mean-pool each group's key/value head projections into one shared head."""
    H, G, Dh = mha.num_heads, num_kv_heads, mha.head_dim
    rep = H // G                                  # heads per group
    gqa = GroupedQueryAttention(mha.q_proj.in_features, H, G)
    gqa.q_proj.weight.data.copy_(mha.q_proj.weight.data)   # unchanged
    gqa.o_proj.weight.data.copy_(mha.o_proj.weight.data)   # unchanged

    def mean_pool(weight):                        # [H*Dh, in] -> [G*Dh, in]
        w = weight.view(G, rep, Dh, -1)            # group the H heads into G groups
        return w.mean(dim=1).reshape(G * Dh, -1)   # average within each group

    gqa.k_proj.weight.data.copy_(mean_pool(mha.k_proj.weight.data))
    gqa.v_proj.weight.data.copy_(mean_pool(mha.v_proj.weight.data))
    return gqa
    # G == H -> rep == 1 -> identity (MHA); G == 1 -> single shared kv head (MQA).


def uptrain(model, pretrain_step_fn, original_steps, alpha=0.05):
    """Adapt the converted checkpoint with ~5% of the original pre-training."""
    for _ in range(int(alpha * original_steps)):
        pretrain_step_fn(model)
    return model
```

GQA is the structural change (grouping the KV heads); uptraining is the cheap
recipe that yields a fast-decoding model from an existing high-quality MHA
checkpoint for a few percent of the original training compute.
