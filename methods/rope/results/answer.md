# Rotary Position Embedding (RoPE)

## Problem

Self-attention is order-blind: with `q, k, v` linear in the token embeddings, the computation is permutation-equivariant, so position must be injected explicitly. Position only matters through the attention logit `q_m^T k_n`. We want that logit to depend on the contents `x_m, x_n` and the **relative offset `m - n`** only; we want position injected as a **per-token transform** of `q` and `k` (so it survives linear attention); and we want a closed form with no learned position parameters and no hard length cap.

## Key idea

Instead of adding a position vector and expanding the dot product, **demand the relative property and solve for the injection function**. Require

```
<f_q(x_m, m), f_k(x_n, n)> = g(x_m, x_n, m - n),   f_q(x, 0) = W_q x,  f_k(x, 0) = W_k x.
```

In 2D, identifying `R^2` with `C` and using `<a, b> = Re[a b*]`, writing `f` in polar form and matching magnitude and phase gives a stable norm-preserving branch where (i) the magnitude is position-independent and (ii) the phase is arithmetic in position. The result is a **rotation by an angle proportional to position**:

```
f_q(x_m, m) = (W_q x_m) e^{i m theta},   f_k(x_n, n) = (W_k x_n) e^{i n theta}
=>  <f_q, f_k> = Re[ (W_q x_m)(W_k x_n)* e^{i(m-n) theta} ]   (depends only on m - n).
```

Lift to dimension `d` (even) by splitting into `d/2` planes and rotating each at its own frequency:

```
f_{q,k}(x_m, m) = R^d_{Theta, m} W_{q,k} x_m,
```

with `R^d_{Theta, m}` block-diagonal, the i-th 2x2 block a rotation by `m * theta_i`, and `theta_i = 10000^{-2(i-1)/d}` (the sinusoidal geometric spectrum reused). Rotations compose, so

```
q_m^T k_n = (R_m W_q x_m)^T (R_n W_k x_n) = x_m^T W_q^T R^d_{Theta, n-m} W_k x_n,
```

The complex form uses `e^{i(m-n)theta}` while the matrix form uses `R_{n-m}` because, for this rotation convention, `q^T R_delta k = Re[q k* e^{-i delta}]`. In both forms, only the relative difference appears.

## Properties

- **Relative by construction**, no learned position table, no clip, no bias bucket; no length cap.
- **Long-range decay envelope.** Group into `d/2` complex pairs; with `h_i = q_{[2i:2i+1]} k_{[2i:2i+1]}*` and partial sums `S_j = sum_{i<j} e^{i(m-n)theta_i}` (`S_0 = 0`, `h_{d/2} = 0`), summation by parts gives `sum_i h_i e^{i(m-n)theta_i} = - sum_i S_{i+1}(h_{i+1}-h_i)`, hence `|sum_i h_i e^{i(m-n)theta_i}| <= (max_i |h_{i+1}-h_i|) sum_i |S_{i+1}|`. The positional envelope `(1/(d/2)) sum_i |S_i|` is a decaying envelope rather than a strict monotone function as `|m - n|` grows.
- **Compatible with linear attention.** Because rotation preserves norm, apply it after the non-negative feature maps: `Attention_m = sum_n (R_m phi(q_m))^T (R_n psi(k_n)) v_n / sum_n phi(q_m)^T psi(k_n)`. Position rides on the per-token features, so the O(N) factorization survives; additive biases that live in the N x N matrix do not fit this factorization.

## Efficient realization

Don't form the sparse matrix. For a vector `x`,

```
R^d_{Theta, m} x = x (*) [cos m*theta_1, cos m*theta_1, cos m*theta_2, cos m*theta_2, ...]
                 + rotate(x) (*) [sin m*theta_1, sin m*theta_1, sin m*theta_2, sin m*theta_2, ...]
```

where `rotate(x) = [-x_2, x_1, -x_4, x_3, ...]` (per-pair 90-degree swap). Two elementwise multiplies and an add: O(d).

## Code

The first pair of helpers mirrors the HuggingFace RoFormer interleaved implementation. The second mirrors the HuggingFace LLaMA split-half `rotate_half` implementation. They are equivalent up to a fixed permutation of the head dimension, but a model must use one layout consistently.

```python
import torch
import torch.nn as nn

def inverse_frequencies(head_dim, base=10000, device=None):
    # one-based theta_i = base^{-2(i-1)/d}; arange(0, d, 2) is the code form.
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for pairwise rotations")
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

def roformer_sinusoidal_pos(positions, head_dim, base=10000):
    inv_freq = inverse_frequencies(head_dim, base, positions.device)
    angles = positions[:, None].float() * inv_freq[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)

def apply_roformer_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
    sinusoidal_pos = sinusoidal_pos.to(device=query_layer.device, dtype=query_layer.dtype)
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
    cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)

    rotate_half_query = torch.stack(
        [-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1
    ).reshape_as(query_layer)
    query_layer = query_layer * cos_pos + rotate_half_query * sin_pos

    rotate_half_key = torch.stack(
        [-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1
    ).reshape_as(key_layer)
    key_layer = key_layer * cos_pos + rotate_half_key * sin_pos

    if value_layer is not None:
        rotate_half_value = torch.stack(
            [-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1
        ).reshape_as(value_layer)
        value_layer = value_layer * cos_pos + rotate_half_value * sin_pos
        return query_layer, key_layer, value_layer
    return query_layer, key_layer

def llama_rotary_tables(position_ids, head_dim, base=10000, dtype=None):
    inv_freq = inverse_frequencies(head_dim, base, position_ids.device)
    inv_freq = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids = position_ids[:, None, :].float()
    freqs = (inv_freq @ position_ids).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    if dtype is not None:
        cos, sin = cos.to(dtype=dtype), sin.to(dtype=dtype)
    return cos, sin

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

def apply_llama_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class PositionStrategy:
    def __init__(self, head_dim, base=10000, layout="llama"):
        if layout not in {"llama", "roformer"}:
            raise ValueError("layout must be 'llama' or 'roformer'")
        self.head_dim = head_dim
        self.base = base
        self.layout = layout

    def apply(self, q, k, positions):
        if self.layout == "roformer":
            sinusoidal_pos = roformer_sinusoidal_pos(positions, self.head_dim, self.base)
            return apply_roformer_rotary_position_embeddings(sinusoidal_pos, q, k)

        position_ids = positions[None, :].expand(q.shape[0], -1)
        cos, sin = llama_rotary_tables(position_ids, self.head_dim, self.base, dtype=q.dtype)
        return apply_llama_rotary_pos_emb(q, k, cos, sin)

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, position):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.position = position

    def forward(self, x, positions=None, mask=None):
        B, T, _ = x.shape
        if positions is None:
            positions = torch.arange(T, device=x.device)
        split = lambda t: t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.Wq(x)), split(self.Wk(x)), split(self.Wv(x))
        q, k = self.position.apply(q, k, positions)
        logits = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            logits = logits.masked_fill(mask, float("-inf"))
        o = logits.softmax(dim=-1) @ v
        return self.Wo(o.transpose(1, 2).reshape(B, T, -1))
position = PositionStrategy(head_dim=64, layout="llama")
attention = SelfAttention(d_model=768, n_heads=12, position=position)
```
