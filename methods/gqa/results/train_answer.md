Autoregressive decoding of large Transformers is limited by memory bandwidth, not raw compute. Each new token forces the accelerator to stream the full model weights and the entire history of cached keys and values across the memory bus, while doing only one token's worth of arithmetic with them. For a layer with batch b, length n, model dimension d, and h heads, the arithmetic is Theta(b n d^2), but the memory traffic from the growing key-value cache alone is Theta(b n^2 d). The resulting memory-to-compute ratio is n/d + 1/b, and the n/d term comes directly from storing a separate key head and value head per position per head. Once the sequence length approaches the model dimension, which is common for long-input generation, the layer is deeply memory-bound and decoding stalls.

The standard multi-head baseline pays this full price for full representational capacity. The opposite extreme, multi-query attention, keeps all query heads but collapses keys and values to a single shared head, cutting the cache by a factor of h. That removes the bandwidth problem but forces every query head through the same key/value subspace, which degrades quality and can make training and fine-tuning unstable. It is also an all-or-nothing switch: there is no knob between the two extremes, and a trained multi-head checkpoint cannot become a multi-query checkpoint without retraining from scratch. So the useful operating point is neither endpoint but somewhere in between.

The method I propose is Grouped-Query Attention, abbreviated GQA. Instead of giving every query head its own key and value, or forcing all query heads to share one, the h query heads are partitioned into G groups, and each group shares a single key head and a single value head. With G = h this recovers ordinary multi-head attention; with G = 1 it recovers multi-query attention; intermediate G interpolates smoothly. The cache now has shape [b, G, n, k] instead of [b, h, n, k], so the offending n/d memory term becomes nG/(dh). A modest choice such as G = 8 cuts the cache by a factor of h/8 while still maintaining several distinct key/value subspaces, which protects model quality and avoids the brittleness of a single shared head. Because the cache grows only linearly with model width while parameters and FLOPs grow quadratically, large models can comfortably afford an intermediate G without surrendering the speed advantage.

GQA is applied to decoder self-attention and cross-attention, where the sequential decode loop reloads the cache every step. Encoder self-attention should remain ordinary multi-head, because the encoder processes its input in parallel and is not bandwidth-bound by a growing decode cache. A second practical benefit is that an existing trained multi-head checkpoint can be converted cheaply. Only the key and value projections change; query and output projections, embeddings, feed-forward layers, and layer norms are copied unchanged. For each group, the shared key projection is the mean of the original h/G key projections in that group, and similarly for values. Mean-pooling preserves the average response of the merged heads and is less destructive than selecting one head or reinitializing randomly. After conversion, a small amount of continued pre-training on the original recipe, typically a few percent of the original step budget, lets the averaged heads re-coordinate with the untouched query and output projections. This avoids the cost of training a new model from scratch.

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
    for _ in range(int(alpha * original_steps)):
        pretrain_step_fn(model)
    return model
```
