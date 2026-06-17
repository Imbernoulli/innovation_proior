# BASED, distilled

BASED is a sub-quadratic sequence mixer that recovers attention-quality recall while keeping a
bounded, *tunable* recurrent state. It combines (1) global **second-order Taylor linear attention** —
linear attention whose feature map makes the kernel approximate the softmax exponential — with (2)
exact softmax attention in **small sliding windows** and short causal convolutions for local
precision. The feature dimension and window size dial the recurrent-state size, so the same design
slides along the recall-memory frontier from cheap-and-forgetful to recall-perfect.

## Problem it solves

Softmax attention recalls perfectly but its generation state — the KV-cache — grows linearly with
sequence length, throttling memory and throughput. Fixed-small-state sub-quadratic mixers (gated
convolutions, SSMs like Mamba) are cheap but lose recall as the number of key-value pairs grows.
The goal: a mixer with a bounded, controllable recurrent state that still does in-context key-value
lookup (MQAR) at long context.

## Why a tunable state, not a tiny one (the lower bound)

Reducing associative recall to the one-way **index problem** (Alice holds x in {0,1}^N, Bob holds i,
output x_i; one-way randomized communication complexity Omega(N), Jayram et al. 2008) shows: *any*
model whose position-i state depends only on the prefix u[0..i-1] needs max_i |state_i| = Omega(N)
bits to solve recall exactly. Alice encodes x as recall pairs (key j, value x_j), runs the model,
ships only its state to Bob, Bob queries i and reads x_i — a one-way protocol, so the state is
Omega(N). Corollary: a fixed-state SSM with O(1)-bit entries needs d + state = Omega(N). So recall
and a small state are fundamentally in tension; the right response is to make state size a knob.

## Key idea

Softmax couples q and k inside exp(q^T k), forcing a growing cache. Replace it with a feature-map
dot product phi(q)^T phi(k); associativity then collapses the over-keys computation into a
fixed-shape running state:

```
S_i = S_{i-1} + phi(k_i) v_i^T   in R^{d~ x d}      (KV-state)
z_i = z_{i-1} + phi(k_i)          in R^{d~}          (normalizer)
y_i = phi(q_i)^T S_i / (phi(q_i)^T z_i)
```

State size scales with the feature dimension d~ — the dial. The feature map must produce *spiky*,
low-entropy weights (like softmax) or recall fails. Use the **2nd-order Taylor expansion of exp**:

```
k(q, k) = phi(q)^T phi(k) = 1 + (q^T k)/sqrt(d~) + (q^T k)^2 / (2 d~) ,
phi(x) = [ 1 ,  x / d~^{1/4} ,  (x ⊗ x) / (sqrt(2) · sqrt(d~)) ]  in R^{1 + d~ + d~^2}.
```

This kernel is (a) realized exactly by a finite, deterministic, parameter-free feature map; (b)
**non-negative**: 1 + s + s^2/2 = ((s+1)^2 + 1)/2 >= 1/2 > 0 for all s; (c) **spiky**: grows
quadratically in q^T k, sharpening weight onto matching keys. The 1/sqrt(d~) temperature keeps q^T k
in the range where the 2-term truncation tracks exp (drop the x^3/6 tail safely). Projecting q, k to
a small d~ (e.g. 16) keeps the d~^2 feature expansion — and the state O(d~^2 d) — cheap, and grows
recall capacity without adding model parameters.

## Quadratic (train) view = recurrent (generation) view

For moderate T, train with the parallel masked-matmul view (tensor-core friendly, no sequential
scan); generate with the recurrent view (O(1)/token, fixed state). They are exactly equal:

```
row_i[ causal_mask(phi(Q) phi(K)^T) · V ]
   = sum_{j<=i} (phi(q_i)^T phi(k_j)) v_j = phi(q_i)^T S_i,   normalizer = phi(q_i)^T z_i.
```

The same kernel can be computed as `1 + s + s^2/2` from pre-scaled raw scores, but the standard
quadratic training view materializes `phi(q)` and `phi(k)`, forms the causal `phi(Q) phi(K)^T`
matrix, and uses the recurrent denominator `phi(q_i)^T z_i`.

## Local precision: small windows + short conv

Global linear attention is blunt at local token shifts/comparisons. BASED adds exact softmax
attention over **small** sliding windows (around 64-128 tokens, sized for tensor-core occupancy, not the
w=4096 of prior LLMs; cache capped at w) and short causal convolutions (filter width 3), which
supply precise local mixing the linear attention cannot form. Linear attention handles long-range
with a large dial-able state; window/conv handle short-range exactness with tiny fixed state.

## Working code (Taylor linear-attention core, quadratic train view)

The global core expands `q,k` through the Taylor feature map, then uses the quadratic masked-matmul
form. Short convolutions and small sliding-window attention are separate local mixers around this
core.

```python
import math
import torch
import torch.nn as nn
from einops import rearrange


class TaylorExp(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.r2 = math.sqrt(2)
        self.rd = math.sqrt(input_dim)          # sqrt(d~)
        self.rrd = math.sqrt(self.rd)           # d~^(1/4)

    def forward(self, x):                        # [B, H, T, d~]
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        ones = torch.ones(x[..., :1].shape, device=x.device, dtype=x.dtype)
        return torch.cat([ones, x / self.rrd, x2 / self.rd], dim=-1)


def repeat_kv(x, n_rep: int):
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class BasedLinearAttention(nn.Module):
    def __init__(self, d_model: int, feature_dim: int = 16, num_heads: int = 12,
                 num_key_value_heads: int | None = None, eps: float = 1e-12):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = d_model // self.num_key_value_heads
        self.feature_map = TaylorExp(feature_dim)
        self.proj_q = nn.Linear(d_model, feature_dim * num_heads, bias=False)
        self.proj_k = nn.Linear(d_model, feature_dim * self.num_key_value_heads, bias=False)
        self.proj_v = nn.Linear(d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.proj_o = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        self.eps = eps

    def forward(self, hidden_states):             # [B, T, d_model]
        b, t, _ = hidden_states.size()
        q = self.proj_q(hidden_states).view(b, t, self.num_heads, self.feature_dim).transpose(1, 2)
        k = self.proj_k(hidden_states).view(b, t, self.num_key_value_heads, self.feature_dim).transpose(1, 2)
        v = self.proj_v(hidden_states).view(b, t, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        q, k = self.feature_map(q), self.feature_map(k)
        causal = torch.tril(torch.ones((t, t), device=q.device, dtype=q.dtype))
        A_qk = torch.einsum("bhnd,bhmd->bhnm", q, k) * causal
        out = torch.einsum("bhnm,bhme->bhne", A_qk.to(hidden_states.dtype), v.to(hidden_states.dtype))
        z = 1 / (torch.einsum("bhld,bhld->bhl", q, k.cumsum(2)) + self.eps)
        y = out * z[..., None]
        y = rearrange(y, "b h l d -> b l (h d)")
        return self.proj_o(y.to(hidden_states.dtype))
```

## Relation to prior methods

- **Softmax attention** — exact recall but O(N) growing KV-cache; BASED keeps the
  spiky-weight behavior (via Taylor-2) but factors the score so the state is fixed-shape and
  dial-able.
- **Linear attention (Katharopoulos et al. 2020)** — same associativity/recurrence; BASED's
  contribution is the Taylor-2 feature map (spiky, non-negative, deterministic, parameter-free)
  in place of elu+1, plus small d~ as the state dial.
- **Random-feature / learned feature maps (Performer; Hedgehog)** — approximate exp via random
  features (variance, many features) or a trained MLP (extra params/stage); Taylor-2 is exact for a
  finite map with neither.
- **Sliding window attention** — exact local recall capped at w; BASED uses small (64-128) windows
  for tensor-core efficiency and pairs them with global linear attention for long range.
- **Mamba / fixed-state SSMs** — best prior recall-per-memory but state cannot be cheaply enlarged
  to attention's recall; the Omega(N) bound shows why, and BASED's feature dimension is the missing
  knob.
