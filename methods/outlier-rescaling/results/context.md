## Research question

Transformer language models trained with pre-norm residual blocks reliably develop a few extreme values. The recurring forms are:

- **Attention sinks**: a small number of tokens, most often the first token, receive very large attention logits and absorb a large fraction of the softmax mass from many queries.
- **Massive activations**: a few coordinates of special tokens reach magnitudes far above the surrounding activations.
- **Fixed-dimension residual outliers**: the same feature dimensions carry persistently large activations across most tokens and inputs.

The central question is what role these outliers serve during training, and whether the attention-side and residual-side phenomena share one mechanism.

## Background

The working architecture is a pre-norm decoder-only transformer. A sequence of hidden states `H_i ∈ R^{L×d}` is updated by residual additions,

`H_{i+1} = H_i + F_i(N_i(H_i))`,

where each sublayer sees a normalized residual stream. In this setting two normalizers matter most.

Softmax attention computes weights

`a_j = exp(z_j) / Σ_k exp(z_k)`,

then returns `Σ_j a_j v_j`. The weights must sum to one, so a head has no literal "attend to nothing" state. If it gives a large logit to a token with a small value vector, that token changes the denominator while contributing little content.

RMSNorm computes

`RMSNorm(x) = λ ⊙ x / sqrt(mean(x²) + eps)`.

It removes LayerNorm's mean subtraction but keeps the cross-dimensional scaling: one large coordinate raises the shared RMS denominator and shrinks all coordinates after division. The post-division weight `λ` controls each coordinate's direct contribution but cannot change the denominator.

Diagnostic observations about existing systems make the analogy hard to ignore:

- Across dense and MoE language models with different attention designs, averaged attention maps show a first-token sink while sorted hidden-state activations show persistent vertical stripes in a few feature dimensions. The token-axis sink and feature-axis stripe have the same shape after sorting by magnitude.
- The residual stripes are largely input-independent: the same dimensions light up across most tokens and prompts, so they are not ordinary input-specific semantic features.
- A model with gated attention has weaker attention sinks and no prominent massive activations, yet still has a persistent large residual dimension. The residual phenomenon survives when the attention-side outliers are suppressed.
- A model with explicit learnable attention sinks has no attention sinks on real tokens and no massive activations. The fixed residual outlier still remains.
- Attention sink tokens have unusually small value-vector norms. They receive large probability mass but contribute little vector content.
- Replacing softmax attention with sigmoid or linear attention reduces attention-side massive activations; replacing RMSNorm with a pointwise function reduces residual outliers. The outliers track the normalizers.
- Outliers predominantly originate in feed-forward blocks and are amplified by the down projection. Swish-gated FFNs create larger activations than sigmoid-gated GLUs.

A measured detail about the residual stripe is that the post-normalization weight `λ_d` on the outlier dimension is unusually small: across trained models most RMSNorm weights sit near one, while the persistent residual-outlier dimension can carry a weight on the order of a few thousandths.

## Baselines

- **StreamingLLM / attention sinks (Xiao et al. 2023).** Sliding-window attention fails unless the first few key-value pairs are retained, because those early tokens act as attention sinks. The work establishes that sinks are functionally load-bearing for long-context inference, but mainly explains them as a consequence of softmax needing to allocate probability mass.

- **Massive activations (Sun et al. 2024).** A few constant, input-agnostic coordinates of special tokens act like implicit biases and can be replaced by explicit learnable attention bias or sink keys.

- **Gated Attention (Bondarenko et al. 2023; Qiu et al. 2025).** A head-wise or element-wise sigmoid gate scales the attention output before `o_proj`. In Qwen3-style code, the gate logits are split out of `q_proj`, reshaped to head-wise or element-wise scores, and applied as `attn_output * sigmoid(gate_score)` after attention and before the output projection. This lets a head reduce its output without parking mass on a sink token.

- **RMSNorm (Zhang and Sennrich 2019).** RMSNorm is the residual-side normalizer used by the transformer stack. Its shared denominator is exactly what allows a fixed large dimension to rescale the other dimensions.

- **Dynamic Tanh / normalization-free transformer variants (Zhu et al. 2025; Chen et al. 2025).** Dynamic Tanh replaces normalization with the pointwise function `tanh(alpha*x)*weight + bias`. No coordinate controls another through a shared denominator.

- **Outlier-feature mitigation by removing normalization (He et al. 2024 and related work).** These approaches identify normalization as a source of outlier features and redesign blocks to avoid them.

- **SwiGLU / GLU variants (Shazeer 2020).** `down(act(gate(x)) * up(x))` is the standard FFN form. Swish and related unbounded activations produce larger intermediate values than sigmoid GLU; this makes the FFN a natural source of residual outliers.

- **Quantization mitigations (Dettmers 2022; SmoothQuant; outlier suppression; Hadamard rotations; NVFP4).** These methods migrate difficulty from activations to weights, clip/suppress extreme channels, or rotate outliers across dimensions. They make inference more robust but accept the trained activations as given.

- **StyleGAN2 demodulation and FiLM/adaLN-style scaling.** Vision models document normalization-driven outliers and use various forms of learned, input-conditioned scaling within or around normalization layers.

## Evaluation settings

The natural yardstick is controlled language-model pretraining plus post-training quantization stress tests.

- **Models.** Dense pre-norm decoder-only transformers in the Llama/Qwen lineage, and larger hybrid MoE variants with alternating softmax and linear attention. The dense setting uses a 2B-scale model with hidden size 2048, 28 layers, RMSNorm, RoPE, grouped-query attention, QK normalization, SwiGLU FFNs, and head dimension 256. Hybrid MoEs add many routed experts and use softmax attention every fourth layer.
- **Training budgets.** Controlled dense ablations use a fixed large token budget; scaling runs use substantially larger MoE budgets. Parameter counts are matched when a structural replacement adds parameters.
- **Architecture axes.** Full, linear, and hybrid attention; attention with or without the existing output gate; RMSNorm versus pointwise Dynamic Tanh; SwiGLU versus sigmoid GLU; residual activation clipping at several thresholds; residual-normalization-slot replacements under matched parameter count.
- **Metrics.** Pretraining loss, training stability, maximum activation magnitude, and downstream accuracy on knowledge, science, math, code, and multilingual benchmarks.
- **Quantization protocol.** SmoothQuant calibration on a held-out sequence set; FP8 E4M3 with block-scaled weights and dynamic per-token activations; FP4/NVFP4 with small weight blocks, hierarchical scaling, and dynamic per-token activation scaling.
- **Efficiency.** End-to-end training overhead in a Megatron-LM/ZeRO-style setup while varying hidden size and tracking any added residual-slot compute.

## Code framework

The harness is a Qwen3-style PyTorch decoder block: RMSNorm before attention, residual add, RMSNorm before the MLP, residual add. Gated attention and Dynamic Tanh are existing components. The open slot is the residual normalization module used at both pre-norm sites.

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


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class ResidualNormSlot(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # TODO: fill the residual normalization slot.
        pass

    def forward(self, hidden_states):
        # TODO: return the transformed hidden states.
        pass


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, norm_cls=ResidualNormSlot):
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
