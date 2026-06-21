The thing I keep running into across trained transformer language models is a small set of extreme values that refuse to be explained away. Feed ordinary sequences through a pre-norm decoder-only stack and two patterns appear together. A few tokens, almost always the first, receive attention logits far larger than the rest, so many queries dump a large share of their softmax mass onto them — the attention sink. And if I sort feature dimensions by mean absolute hidden-state magnitude, a few dimensions form a vertical stripe that is large for most tokens, across most prompts, in the same coordinates — the fixed residual outlier, plus the related massive activations on special tokens. After sorting by magnitude the token-axis sink and the feature-axis stripe have the same shape; they look like one phenomenon viewed along different axes. These values matter in practice because they dominate the activation dynamic range and wreck low-bit quantization: W8A8 already wastes range on them and W4A4 is brittle. The trouble is that every obvious fix backfires. Clipping the large activations destabilizes training or costs quality; replacing the normalizer that seems to manufacture them also hurts. So I will not start by declaring the outliers harmful. I want to know what computation disappears when they are gone.

The attention sink gives the first clean reading. The sink token receives huge attention weight, yet its value vector has unusually small norm — large probability mass, almost no content. Let the real-token logits be $z_i$ with values $v_i$, the sink logit be $z_s$ with value $v_s$, and write $S=\sum_i e^{z_i}$ and $w_s = e^{z_s}/(e^{z_s}+S)$. Then every real-token weight is $w_i = e^{z_i}/(e^{z_s}+S) = (1-w_s)\,e^{z_i}/S$, so the attention output factors as
$$\sum_i w_i v_i + w_s v_s = (1-w_s)\sum_i \mathrm{softmax}(z)_i\, v_i + w_s v_s.$$
When $\lVert v_s\rVert$ is small, the sink mostly donates denominator mass and rescales the genuine real-token mixture by about $1-w_s$. The sink is not adding information; it is a scale knob built out of softmax's normalization, the one knob a sum-to-one operation otherwise lacks since there is no literal "attend to nothing" state. That diagnosis is corroborated by the way attention sinks shrink when softmax token-mixing is swapped for sigmoid or linear attention: with no shared denominator to drive, the token-side sink loses its purpose.

The same logic transfers to the residual stripe through RMSNorm, which divides every coordinate by the shared $\sqrt{\mathrm{mean}(x^2)+\epsilon}$ and then applies a learned post-division weight $\lambda$. One huge coordinate raises that shared denominator and shrinks every coordinate after the division; the matching fingerprint, if the outlier dimension is not meant to contribute directly, is a tiny post-norm weight $\lambda_d$ on that same dimension. That is exactly what the trained models show — most RMSNorm weights sit near one, while the persistent residual-outlier dimension carries a weight on the order of a few thousandths. The network makes the coordinate large before the denominator and mutes it immediately after. To pin the magnitude claim down honestly I keep the factor of $D$ explicit and drop $\epsilon$ only to tighten an upper bound (including it makes the denominator larger, so the bound still holds). Take $h\in\mathbb{R}^D$, outlier dimension $d$, define the outlier norm fraction $r=|h_d|/\lVert h\rVert_2$ and the unit vector $u=h/\lVert h\rVert_2$ with $|u_d|=r$ and $\sum_{i\neq d}u_i^2=1-r^2$, and assume the observed suppression $|\lambda_d|\le \varepsilon\lVert\lambda\rVert_\infty$ with $\varepsilon<1$. Since $\lVert h\rVert_{\mathrm{rms}}=\lVert h\rVert_2/\sqrt{D}$, the RMS norm after normalization is exactly $\lVert\lambda\odot u\rVert_2$, and
$$\lVert\lambda\odot u\rVert_2^2 = \lambda_d^2 r^2 + \sum_{i\neq d}\lambda_i^2 u_i^2 \le \lVert\lambda_{-d}\rVert_\infty^2(1-r^2)+\lambda_d^2 r^2.$$
Relaxing $\lVert\lambda_{-d}\rVert_\infty\le\lVert\lambda\rVert_\infty$ and $\lambda_d^2\le\varepsilon^2\lVert\lambda\rVert_\infty^2$ gives
$$\lVert\mathrm{RMSNorm}(h)\rVert_{\mathrm{rms}} \le \lVert\lambda\rVert_\infty\sqrt{(1-r^2)+\varepsilon^2 r^2} = \lVert\lambda\rVert_\infty\sqrt{1-(1-\varepsilon^2)r^2}.$$
Because $\varepsilon<1$, this bound decreases as $r$ grows: a larger residual outlier shrinks the post-normalization feature magnitude of everything else, while its own dimension is silenced by the small $\lambda_d$. The attention sink and the residual sink are therefore the same algebraic move — a large pre-normalization component controlling the scale of the non-outlier components through a shared denominator. This unifies the rest of the diagnostics: replacing RMSNorm with the pointwise Dynamic Tanh $\tanh(\alpha x)\,\mathrm{weight}+\mathrm{bias}$ removes the residual outlier but also removes the cross-coordinate denominator, which is why it only trains at a much smaller learning rate; clipping the residual activation while keeping RMSNorm leaves the denominator in place but forbids driving it, and the resulting loss spikes say the large value and the normalizer act as a unit; and SwiGLU's unbounded Swish is a natural outlier factory that a bounded sigmoid GLU starves, which is why sigmoid GLU pays a quality cost. The outliers are not the goal — the rescaling they induce is.

So the goal becomes precise: keep the rescaling, but remove the need to store its signal as an enormous activation. I propose two residual-normalization modules that do this, which I will call PreAffine and GatedNorm (the gated-RMSNorm form). PreAffine attacks the fixed, input-independent residual stripe. Because that stripe reuses the same dimensions across tokens and inputs, a static per-dimension parameter is a plausible carrier, so I multiply by a learned $\lambda_1$ before normalizing:
$$\mathrm{PreAffineRMSNorm}(x) = \mathrm{RMSNorm}(\lambda_1 \odot x).$$
The placement is the whole point. The ordinary RMSNorm weight sits after the division and can only scale a coordinate's direct contribution; $\lambda_1$ sits before the division, so a large $\lambda_1^{(d)}$ can inflate the RMS denominator in dimension $d$ — and thus rescale the other dimensions — even when the residual activation itself is ordinary. That relocates the denominator-control trick from activation space into parameter space. PreAffine still leaves a large internal value inside $\lambda_1\odot x$, however, so if I want the computation to depend on no large value at all, the model needs a direct post-normalization scale path. Gated attention already supplies the template — it lets a head reduce its output without parking mass on a sink token — and the residual-side analogue takes $y=\mathrm{RMSNorm}(x)$ and multiplies it by a learned self-gate. The gate has three requirements, each settled against the obvious alternative. It must be input-dependent, because a fixed scale would only reproduce the average effect rather than the per-token rescaling the sink performed. It must be cheap, because it sits at every normalization site, which I get with a low-rank bottleneck whose cost is linear in hidden size with a small constant. And it must act element-wise, because a single scalar per token cannot decide which individual coordinates to damp. That gives GatedNorm:
$$y = \mathrm{RMSNorm}(x), \qquad g = \sigma\!\big(W_{\mathrm{up}}\,\mathrm{silu}(W_{\mathrm{down}}\,y)\big), \qquad y' = g \odot y,$$
with $W_{\mathrm{down}}$ projecting $d\to r$ and $W_{\mathrm{up}}$ projecting $r\to d$ at a small rank $r$. The sigmoid is load-bearing: this path is meant to be a bounded down-scaling path, and $\tanh$ can flip signs while SiLU or identity can amplify without limit, which would re-create the very large activations I am trying to eliminate. With the bounded gate the model shrinks a component directly instead of manufacturing a residual sink to shrink it indirectly through the RMS denominator. Concretely this also predicts that, once the explicit gate supplies residual rescaling, the choices that mattered mainly because they helped fabricate outliers — Swish over sigmoid GLU, RMSNorm over Dynamic Tanh — should matter less, and low-bit quantization should get easier because no coordinate has to carry an enormous scale signal anymore.

I write the modules in the same shape as the Qwen3-style block I am modifying — RMSNorm before attention, gated attention that splits gate logits out of `q_proj` and applies `sigmoid(gate_score)` before `o_proj`, RMSNorm before the MLP — with the residual normalization slot filled by either `PreAffineRMSNorm` or `GatedRMSNorm`.

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
