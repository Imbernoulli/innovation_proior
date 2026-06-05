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
is then $\propto G$, and the memory ratio term is $nG/(dH)$:

- $G = 1$ recovers MQA (one shared KV head);
- $G = H$ recovers multi-head attention (MHA);
- intermediate $G$ interpolates — capacity spread over $G$ subspaces instead of
  one, cache cut by a factor $H/G$.

Because the KV cache scales more slowly than the model's $d^2$ compute and
parameter cost, large models can afford an intermediate $G$ without giving back
the full MHA cache. A modest target such as $G=8$ keeps several key/value
subspaces, still cuts the cache by $H/8$, and avoids the tensor-parallel waste
of replicating a single shared head. GQA is applied to decoder self-attention
and cross-attention, not encoder self-attention, because the encoder runs in
parallel and does not stream a growing decode cache.

## Uptraining (cheap conversion from an existing checkpoint)

Don't train from scratch — convert a trained MHA checkpoint in two steps:

1. **Convert.** For each group, build the shared key/value projection by
   **mean-pooling** the projection matrices of the $H/G$ heads in that group:
   $$W_k^{g} = \tfrac{1}{|g|}\sum_{j\in g} W_k^{(j)}, \qquad
     W_v^{g} = \tfrac{1}{|g|}\sum_{j\in g} W_v^{(j)}.$$
   Mean-pooling preserves the group's average response and is less destructive
   than selecting one head or random initialization. Query and output
   projections copy over unchanged.
2. **Adapt.** Continue pre-training on the original recipe/data for a small
   fraction $\alpha$ of the original steps, with $\alpha \approx 0.05$ as the
   cheap adaptation budget.

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
  k, v = repeat_key_value_heads(k, H/G), repeat_key_value_heads(v, H/G)
  out = o_proj( softmax(q k^T / sqrt(head_dim)) v )
```

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionConfig:
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_key_value_heads=None,
        attention_dropout=0.0,
        attention_bias=False,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_key_value_heads = (
            num_heads if num_key_value_heads is None else num_key_value_heads
        )
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_heads must be divisible by num_key_value_heads")
        self.num_key_value_groups = num_heads // self.num_key_value_heads
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias


def repeat_key_value_heads(hidden_states, n_rep):
    """Expand cached key/value heads from [b, G, s, k] to [b, H, s, k]."""
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, seq_len, head_dim
    )


class DecoderAttention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=config.attention_bias
        )

    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx
            )

        key_states = repeat_key_value_heads(key_states, self.num_key_value_groups)
        value_states = repeat_key_value_heads(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)


def convert_attention_checkpoint(pretrained_attention, config):
    """Mean-pool each group's key/value projections into one shared head."""
    converted = DecoderAttention(
        config, layer_idx=getattr(pretrained_attention, "layer_idx", 0)
    )
    source_heads = pretrained_attention.num_heads
    target_heads = config.num_key_value_heads
    head_dim = pretrained_attention.head_dim
    if source_heads % target_heads != 0:
        raise ValueError("source heads must be divisible by target key/value heads")
    heads_per_group = source_heads // target_heads

    def copy_linear(dst, src):
        dst.weight.copy_(src.weight)
        if dst.bias is not None and src.bias is not None:
            dst.bias.copy_(src.bias)

    def mean_pool_weight(weight):
        grouped = weight.view(target_heads, heads_per_group, head_dim, -1)
        return grouped.mean(dim=1).reshape(target_heads * head_dim, -1)

    def mean_pool_bias(bias):
        grouped = bias.view(target_heads, heads_per_group, head_dim)
        return grouped.mean(dim=1).reshape(target_heads * head_dim)

    with torch.no_grad():
        copy_linear(converted.q_proj, pretrained_attention.q_proj)
        copy_linear(converted.o_proj, pretrained_attention.o_proj)
        converted.k_proj.weight.copy_(mean_pool_weight(pretrained_attention.k_proj.weight))
        converted.v_proj.weight.copy_(mean_pool_weight(pretrained_attention.v_proj.weight))
        if converted.k_proj.bias is not None and pretrained_attention.k_proj.bias is not None:
            converted.k_proj.bias.copy_(mean_pool_bias(pretrained_attention.k_proj.bias))
        if converted.v_proj.bias is not None and pretrained_attention.v_proj.bias is not None:
            converted.v_proj.bias.copy_(mean_pool_bias(pretrained_attention.v_proj.bias))
    return converted


def continue_pretraining(model, pretrain_step_fn, original_steps, alpha=0.05):
    """Adapt the converted checkpoint with a small continuation run."""
    for _ in range(int(alpha * original_steps)):
        pretrain_step_fn(model)
    return model
```

GQA is the structural change: fewer stored key/value heads, repeated after cache
loading to serve all query heads. Uptraining is the conversion recipe that lets
an existing MHA checkpoint adapt after the key/value projection merge.
