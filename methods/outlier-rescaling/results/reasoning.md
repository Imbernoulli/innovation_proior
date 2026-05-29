OK, let me start from the thing I keep seeing and cannot explain. I take trained transformer language models with different attention designs, feed in ordinary sequences, and record attention maps and residual-stream activations layer by layer. Two patterns keep appearing. A few tokens, usually the first token, receive attention logits much larger than the rest, so many queries send a large part of their softmax mass there. At the same time, if I sort feature dimensions by mean absolute hidden-state magnitude, a few dimensions form a vertical stripe: they are large for most tokens, across many inputs, in the same coordinates. The token-axis sink and the feature-axis stripe look like the same shape rotated into different axes.

The engineering problem is immediate. These values dominate quantization scales. W8A8 already has to waste range on them, and W4A4 is even more brittle. But the obvious fixes are suspicious. Clipping large activations can make the model unstable or poor; replacing the normalizer that seems to create the outlier can also hurt. So I do not want to start by declaring the outliers bad. I want to know what computation becomes unavailable when I remove them.

The attention sink gives me the first clean clue. The sink token gets a huge attention weight, yet its value vector has unusually small norm. If attention weight meant "fetch this token's content," that would be strange. Let the real-token logits be `z_i`, their values be `v_i`, the sink logit be `z_s`, and the sink value be `v_s`. Write `S = Σ_i exp(z_i)` and `w_s = exp(z_s)/(exp(z_s)+S)`. Then every real-token weight is

`w_i = exp(z_i)/(exp(z_s)+S) = (1-w_s) exp(z_i)/S`.

So the attention output is

`Σ_i w_i v_i + w_s v_s = (1-w_s)Σ_i softmax(z)_i v_i + w_s v_s`.

If `v_s` is small, the sink mostly contributes denominator mass. The useful content is the same real-token mixture, scaled down by about `1-w_s`. The sink is not there to add information; it is a scale knob created through softmax's denominator.

Now I can ask whether the fixed residual stripe is the same trick through RMSNorm. RMSNorm divides every coordinate by `sqrt(mean(x^2)+eps)` and then applies a learned weight `λ`. One huge coordinate raises the shared denominator and shrinks all coordinates after the division. If that coordinate is not supposed to contribute directly, the matching fingerprint should be a tiny post-normalization weight on that same dimension. That is exactly what shows up: most RMSNorm weights are near one, while the residual-sink dimension can have a weight around a few thousandths. The network makes the coordinate large before the denominator, then suppresses it immediately after the denominator. That is the residual analogue of a sink token with a small value vector.

Let me check the magnitude claim without hiding a factor of `D`. I will ignore `eps` for the algebra; with `eps` included, the denominator is only larger, so the bound I get remains an upper bound. Take `h ∈ R^D`, one outlier dimension `d`, and a post-norm weight vector `λ`. Let `r = |h_d|/||h||_2`, and assume the observed suppression is true in the form `|λ_d| ≤ ε||λ||_∞` with `ε < 1`. Define `u = h/||h||_2`, so `|u_d| = r` and `Σ_{i≠d}u_i^2 = 1-r^2`. Since `||h||_rms = ||h||_2/√D`, the L2 norm after RMSNorm is `√D ||λ⊙u||_2`, and the RMS norm after RMSNorm is exactly `||λ⊙u||_2`. Now

`||λ⊙u||_2^2 = λ_d^2r^2 + Σ_{i≠d} λ_i^2u_i^2`.

The non-outlier part is at most

`Σ_{i≠d} λ_i^2u_i^2 ≤ ||λ_{-d}||_∞^2 Σ_{i≠d}u_i^2 = ||λ_{-d}||_∞^2(1-r^2)`,

so the tight intermediate bound is

`||RMSNorm(h)||_rms ≤ sqrt(||λ_{-d}||_∞^2(1-r^2) + λ_d^2r^2)`.

Relaxing `||λ_{-d}||_∞ ≤ ||λ||_∞` and `λ_d^2 ≤ ε^2||λ||_∞^2` gives

`||RMSNorm(h)||_rms ≤ ||λ||_∞ sqrt((1-r^2)+ε^2r^2) = ||λ||_∞ sqrt(1-(1-ε^2)r^2)`.

Because `ε<1`, that upper bound decreases as `r` grows. A larger residual outlier can shrink the post-normalization feature magnitude, while its own dimension is muted by the small `λ_d`. The attention sink and the residual sink now have the same algebraic role: a large pre-normalization component controls the scale of non-outlier components through a shared denominator.

This picture makes several diagnostics fall into place. If I remove softmax from token mixing with sigmoid or linear attention, attention-side massive activations shrink because the token-mixing denominator no longer needs a sink. But a residual sink can remain, because RMSNorm is still present. If I replace RMSNorm with Dynamic Tanh, `tanh(alpha*x)*weight+bias`, the operation is pointwise. Each coordinate only sees itself. That removes the cross-dimensional denominator, and the residual outlier mostly vanishes, but the model has also lost the ability for one coordinate to rescale the rest. The instability at ordinary learning rates and the need for a much smaller learning rate are exactly what I would expect if the outlier was carrying a useful scaling function.

Clipping separates the two pieces more cleanly. If RMSNorm is still there and I only cap the residual activation, then the denominator exists but the model cannot drive it with a large coordinate. When clipping at low thresholds causes divergence, and clipping at a high threshold still creates loss spikes, the message is not merely "normalization matters." The large value and the normalizer are acting as a unit. In a model whose attention side already has gated attention, aggressive residual clipping is less catastrophic, which tells me the attention sink was the primary stability hazard; the remaining degradation still says the residual sink has its own job.

The FFN source of the outlier also makes more sense now. A SwiGLU block computes `down(silu(gate(x)) * up(x))`. Swish is unbounded on the positive side, so it can generate large intermediate values that the down projection can amplify into residual outliers. A sigmoid GLU caps the gate in `(0,1)` and therefore starves that route. If the outlier is useful for rescaling, sigmoid GLU should suppress outliers and pay a quality cost even if the rest of the architecture is unchanged. That is the pattern I see. Swish is not only a better nonlinearity in the abstract; in this training regime it gives the model more room to manufacture the scale knob it wants.

Now the goal changes. I do not want to remove the rescaling. I want to remove the need to store the rescaling signal as a huge activation. The first route is to put the fixed residual sink into a parameter. The residual stripe uses the same dimensions across tokens and inputs, so a static per-dimension vector is a plausible carrier. If I multiply by a learned `λ1` before computing RMSNorm,

`PreAffineRMSNorm(x) = RMSNorm(λ1 ⊙ x)`,

then a large `λ1_d` can make the normalizer input large in dimension `d` even when the residual activation itself is ordinary. This is not the same as the usual RMSNorm weight. The usual `λ` sits after the division; it can scale a coordinate's direct contribution but cannot affect the denominator. `λ1` sits before the division, so it can change the RMS and rescale other dimensions. This relocates the outlier-driven denominator control from activation space into parameter space.

That still leaves a large internal value inside `λ1⊙x`. If I want the computation to stop depending on any large value, the model needs a direct scale path after normalization. Attention already has the template: gated attention lets a head reduce its output without donating mass to a sink. The residual-side version should take `y = RMSNorm(x)` and multiply it by a learned gate `g(y)`. The gate has to be input-dependent, because a fixed scale would only replace the average effect. It has to be cheap, because it appears at every normalization site. It has to address dimensions separately, because a single scalar per token cannot decide which coordinates should be damped.

A low-rank self-gate fits those constraints: down-project `y` from `d` to a small rank `r`, apply a nonlinearity, project back to `d`, and pass through a sigmoid:

`g = sigmoid(W_up(silu(W_down(y))))`, `y' = g ⊙ y`.

The low rank keeps the cost linear in hidden size with a small constant. The element-wise output gives each coordinate its own scale. The sigmoid matters because the path is meant to be a bounded down-scaling path. Tanh can flip signs, SiLU and identity can amplify without bound, and an unbounded gate risks recreating the very large activations I am trying to avoid. With a sigmoid gate, the model can shrink a component directly instead of inventing a residual sink to shrink it indirectly through the RMS denominator.

If the explicit gate supplies the residual rescaling function, then choices that were useful mainly because they helped manufacture outliers should become less important. Dynamic Tanh with a residual gate should recover part of the missing cross-coordinate scale path. Sigmoid GLU should no longer be starved of rescaling capacity just because it makes smaller FFN activations. Low-bit quantization should also be easier because the residual stream no longer needs a few coordinates to carry enormous scale signals.

Let me write the modules in the same shape as the Qwen3-style code I am modifying: RMSNorm before attention, gated attention that splits gate logits out of `q_proj` and applies `sigmoid(gate_score)` before `o_proj`, RMSNorm before the MLP, and the residual normalization slot filled either by a pre-affine RMSNorm or by the explicit gated RMSNorm.

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

Tracing the chain back, the sink denominator algebra tells me an attention sink with a small value vector scales real-token content instead of adding content. The RMSNorm bound gives the residual version: a large coordinate with a tiny post-norm weight can shrink the rest through the shared RMS denominator. Removing normalizers removes both outliers and that scale path; clipping keeps the normalizer but blocks the value it was using. So I keep the scale path and change where it lives: PreAffine moves the denominator control into a pre-RMS parameter, while GatedRMSNorm replaces the indirect denominator trick with a direct bounded element-wise gate.
