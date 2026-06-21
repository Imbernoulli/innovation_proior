The core tension is between recall quality and generation efficiency. Softmax attention gives perfect in-context lookup because it forms a sharp, normalized similarity over all past keys and values, but that comes at the cost of a KV-cache that grows linearly with sequence length, which eventually dominates memory and per-token latency. Sub-quadratic alternatives avoid this by compressing the past into a fixed-size recurrent state, but they tend to fall short on associative recall: plain linear attention with generic feature maps produces flat, high-entropy weights that dilute the matching value, while fixed-state SSMs must pack every binding into a single hidden vector and degrade as the number of key-value pairs grows. So the real need is not a tiny fixed state, but a state whose size is a controllable dial that can trade cost against recall capacity.

The method I propose is BASED. It keeps the factored structure of linear attention, which lets the over-keys computation collapse into a fixed-shape running state, but replaces the usual gentle feature map with one that mimics the softmax exponential. The key ingredient is a second-order Taylor kernel: k(q, k) = 1 + (q^T k) / sqrt(d~) + (q^T k)^2 / (2 d~). This kernel equals phi(q)^T phi(k) for an explicit finite feature map phi(x) = [1, x / d~^{1/4}, (x ⊗ x) / (sqrt(2) sqrt(d~))], so it is deterministic, parameter-free, and requires no random features or learned approximations. It is also strictly positive, since 1 + s + s^2/2 = ((s + 1)^2 + 1) / 2 ≥ 1/2, and it grows quadratically in the dot product, giving the spiky, low-entropy weight distribution that recall needs. The q and k are projected down to a small feature dimension d~ (for example 16) before applying the map, which keeps the d~^2 feature expansion cheap and acts as the state-size dial. Because the kernel is factored, the generation view can maintain a recurrent KV-state S_i and normalizer z_i of fixed shape rather than a growing cache, giving O(1) per-token cost with a bounded state that grows only with the chosen feature dimension.

BASED pairs this global spiky linear attention with two cheap local mechanisms. Global linear attention is powerful for long-range lookup but blunt at fine local token-to-token comparisons, so the mixer also uses exact softmax attention over small sliding windows (tuned for tensor-core occupancy, around 64 to 128 tokens) and short causal depthwise convolutions (filter width 3) to supply precise local shifts. The small window has a cache capped at its width, and the convolution carries almost no state, so neither reintroduces a growing cache. During training, the quadratic masked-matmul view is used: the T × T score matrix is materialized from phi(Q) phi(K)^T, a causal mask is applied, and the same recurrent denominator phi(q_i)^T z_i normalizes each row. This is term-for-term identical to the recurrent generation view, but it maps efficiently to batched matrix multiplication on a GPU. The result is a single architecture whose feature dimension and window size slide it along the recall-memory frontier, from cheap and forgetful all the way toward recall-perfect behavior.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaylorExp(nn.Module):
    """Second-order Taylor feature map realizing 1 + (q^T k)/sqrt(d) + (q^T k)^2/(2d)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.r2 = math.sqrt(2.0)
        self.rd = math.sqrt(input_dim)          # sqrt(d~)
        self.rrd = math.sqrt(self.rd)           # d~^(1/4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, H, T, d~]
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        ones = torch.ones_like(x[..., :1])
        return torch.cat([ones, x / self.rrd, x2 / self.rd], dim=-1)


class BASED(nn.Module):
    """Taylor-2 linear attention + short causal convolution.

    The quadratic training view materialises the T x T causal kernel matrix
    phi(Q) phi(K)^T and normalises by the recurrent denominator phi(q_i)^T z_i.
    At generation time the equivalent recurrent view carries only S_i and z_i.
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int,
        feature_dim: int = 16,
        num_heads: int = 4,
        conv_kernel: int = 3,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.eps = eps

        self.feature_map = TaylorExp(feature_dim)
        self.q_proj = nn.Linear(d_model, feature_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(d_model, feature_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.local_conv = nn.Conv1d(
            d_model, d_model, kernel_size=conv_kernel, groups=d_model, bias=True
        )
        self.conv_pad = conv_kernel - 1

    def _short_conv(self, x: torch.Tensor) -> torch.Tensor:
        h = F.pad(x.transpose(1, 2), (self.conv_pad, 0))
        return self.local_conv(h).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        local = self._short_conv(x)
        h = x + local

        q = self.q_proj(h).view(B, T, self.num_heads, self.feature_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.num_heads, self.feature_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.feature_map(q), self.feature_map(k)      # [B, H, T, 1 + d~ + d~^2]

        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=x.dtype))
        A_qk = torch.einsum("bhqd,bhkd->bhqk", q, k) * causal
        out = torch.einsum("bhqk,bhkd->bhqd", A_qk, v)

        # Recurrent denominator: phi(q_i)^T sum_{j<=i} phi(k_j)
        z = torch.einsum("bhqd,bhqd->bhq", q, k.cumsum(dim=2)).clamp_min(self.eps)
        y = out / z.unsqueeze(-1)

        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(y) + local
```
