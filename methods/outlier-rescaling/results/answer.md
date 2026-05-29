# Outlier-Driven Rescaling: PreAffine and GatedNorm

## Problem

Transformer LLMs grow attention sinks, massive activations, and fixed-dimension residual outliers. These values hurt low-bit quantization, yet clipping them or replacing the normalization that creates them also hurts training. The useful function is not the large value itself; it is the rescaling that the large value induces through a normalizer.

## Key idea

For an attention sink, let real-token logits be `z_i`, values be `v_i`, sink logit be `z_s`, and sink value be `v_s`. With `S = Σ_i exp(z_i)` and `w_s = exp(z_s)/(exp(z_s)+S)`,

`w_i = exp(z_i)/(exp(z_s)+S) = (1-w_s) exp(z_i)/S`,

so the attention output is

`Σ_i w_i v_i + w_s v_s = (1-w_s)Σ_i softmax(z)_i v_i + w_s v_s`.

When `v_s` has small norm, the sink mostly donates denominator mass and scales the real-token output by about `1-w_s`.

For a residual sink, RMSNorm gives the same kind of denominator control across feature dimensions. Ignoring `eps` for the algebra, and noting that `eps` only tightens the upper bound, take `h ∈ R^D`, an outlier dimension `d`, `r = |h_d|/||h||_2`, and `|λ_d| ≤ ε||λ||_∞` with `ε < 1`. With `u=h/||h||_2`,

```text
||RMSNorm(h)||_rms
  = ||λ⊙u||_2
  ≤ sqrt(||λ_{-d}||_∞²(1-r²) + λ_d²r²)
  ≤ ||λ||_∞ sqrt(1 - (1-ε²)r²).
```

The upper bound decreases as the outlier norm fraction `r` grows, while the small post-norm weight `λ_d` suppresses the outlier dimension's direct contribution. Attention sinks and residual sinks are therefore two instances of outlier-driven rescaling.

PreAffine keeps the residual stream small by moving the large denominator factor into a parameter:

`PreAffineRMSNorm(x) = RMSNorm(λ1 ⊙ x)`.

GatedNorm removes the need for a large denominator factor by giving the normalized vector an explicit low-rank, element-wise sigmoid gate:

`y = RMSNorm(x)`, `g = sigmoid(W_up(silu(W_down(y))))`, `y' = g ⊙ y`.

The gate is low-rank for cost, element-wise for per-dimension control, and sigmoid-bounded so the rescaling path is a stable down-scaling path rather than another source of unbounded activations.

## Final modules (PyTorch)

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.layers import LayerNorm2d
except ImportError:
    LayerNorm2d = ()


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            return x * self.weight + self.bias
        return x * self.weight[:, None, None] + self.bias[:, None, None]

    def extra_repr(self):
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"
        )


def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    return module_output


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3GatedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.use_qk_norm = config.use_qk_norm
        self.headwise_attn_output_gate = config.headwise_attn_output_gate
        self.elementwise_attn_output_gate = config.elementwise_attn_output_gate
        self.inv_sqrt_head_dim = 1.0 / math.sqrt(self.head_dim)

        if self.headwise_attn_output_gate:
            q_out = self.num_heads * self.head_dim + self.num_heads
        elif self.elementwise_attn_output_gate:
            q_out = self.num_heads * self.head_dim * 2
        else:
            q_out = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, q_out, bias=config.qkv_bias)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.qkv_bias)
        if self.use_qk_norm:
            self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        batch, seq_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.headwise_attn_output_gate:
            query_states = query_states.view(batch, seq_len, self.num_key_value_heads, -1)
            query_states, gate_score = torch.split(
                query_states,
                [self.head_dim * self.num_key_value_groups, self.num_key_value_groups],
                dim=-1,
            )
            gate_score = gate_score.reshape(batch, seq_len, -1, 1)
            query_states = query_states.reshape(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        elif self.elementwise_attn_output_gate:
            query_states = query_states.view(batch, seq_len, self.num_key_value_heads, -1)
            query_states, gate_score = torch.split(
                query_states,
                [self.head_dim * self.num_key_value_groups] * 2,
                dim=-1,
            )
            gate_score = gate_score.reshape(batch, seq_len, -1, self.head_dim)
            query_states = query_states.reshape(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        else:
            query_states = query_states.view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
            gate_score = None

        key_states = key_states.view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.inv_sqrt_head_dim
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()

        if gate_score is not None:
            attn_output = attn_output * torch.sigmoid(gate_score)

        attn_output = attn_output.reshape(batch, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class PreAffineRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(hidden_size))
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = (self.lambda1 * hidden_states).to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class GatedRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, rank=16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.down_proj = nn.Linear(hidden_size, rank, bias=False)
        self.up_proj = nn.Linear(rank, hidden_size, bias=False)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        y = self.weight * hidden_states.to(input_dtype)
        gate = torch.sigmoid(self.up_proj(F.silu(self.down_proj(y))))
        return gate * y


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, norm_cls=GatedRMSNorm):
        super().__init__()
        self.self_attn = Qwen3GatedAttention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
```
