The core tension is between recall quality and generation efficiency. Softmax attention gives perfect in-context lookup because it forms a sharp, normalized similarity over all past keys and values, but that comes at the cost of a KV-cache that grows linearly with sequence length, which eventually dominates memory and per-token latency. Sub-quadratic alternatives avoid this by compressing the past into a fixed-size recurrent state, but they tend to fall short on associative recall: plain linear attention with generic feature maps produces flat, high-entropy weights that dilute the matching value, while fixed-state SSMs must pack every binding into a single hidden vector and degrade as the number of key-value pairs grows. So the real need is not a tiny fixed state, but a state whose size is a controllable dial that can trade cost against recall capacity.

The method I propose is BASED. It keeps the factored structure of linear attention, which lets the over-keys computation collapse into a fixed-shape running state, but replaces the usual gentle feature map with one that mimics the softmax exponential. The key ingredient is a second-order Taylor kernel: k(q, k) = 1 + (q^T k) / sqrt(d~) + (q^T k)^2 / (2 d~). This kernel equals phi(q)^T phi(k) for an explicit finite feature map phi(x) = [1, x / d~^{1/4}, (x ⊗ x) / (sqrt(2) sqrt(d~))], so it is deterministic, parameter-free, and requires no random features or learned approximations. It is also strictly positive, since 1 + s + s^2/2 = ((s + 1)^2 + 1) / 2 ≥ 1/2, and it grows quadratically in the dot product, giving the spiky, low-entropy weight distribution that recall needs. The q and k are projected down to a small feature dimension d~ (for example 16) before applying the map, which keeps the d~^2 feature expansion cheap and acts as the state-size dial. Because the kernel is factored, the generation view can maintain a recurrent KV-state S_i and normalizer z_i of fixed shape rather than a growing cache, giving O(1) per-token cost with a bounded state that grows only with the chosen feature dimension.

BASED pairs this global spiky linear attention with two cheap local mechanisms. Global linear attention is powerful for long-range lookup but blunt at fine local token-to-token comparisons, so the mixer also uses exact softmax attention over small sliding windows (tuned for tensor-core occupancy, around 64 to 128 tokens) and short causal depthwise convolutions (filter width 3) to supply precise local shifts. The small window has a cache capped at its width, and the convolution carries almost no state, so neither reintroduces a growing cache. During training, the quadratic masked-matmul view is used: the T × T score matrix is materialized from phi(Q) phi(K)^T, a causal mask is applied, and the same recurrent denominator phi(q_i)^T z_i normalizes each row. This is term-for-term identical to the recurrent generation view, but it maps efficiently to batched matrix multiplication on a GPU. The result is a single architecture whose feature dimension and window size slide it along the recall-memory frontier, from cheap and forgetful all the way toward recall-perfect behavior.

What follows is that global Taylor linear-attention core — the feature map, the query/key/value projections, and the quadratic masked-matmul view with its recurrent denominator; the short convolution and the windowed attention are separate local mixers wrapped around this core and are left out of the snippet for compactness. The projections also support grouped key/value heads (`num_key_value_heads`), repeating each key/value head across its query group before the same kernel is applied, so the same core scales down to fewer KV projections without changing the feature map or the recurrence.

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
