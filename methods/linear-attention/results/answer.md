# Linear Attention

## Result

Replace softmax attention by a normalized kernel attention whose similarity factors through a finite feature map:

$$
\operatorname{sim}(q,k)=\phi(q)^T\phi(k),\qquad \phi(x)=\operatorname{elu}(x)+1 .
$$

Then

$$
V'_i =
\frac{\phi(Q_i)^T \sum_j \phi(K_j)V_j^T}
     {\phi(Q_i)^T \sum_j \phi(K_j)} ,
$$

so the key/value matrix and key normalizer are computed once instead of forming an `N x N` attention matrix. With feature dimension `C` and value dimension `M`, the attention computation is `O(NCM)` in sequence length.

For causal attention, the sums become prefixes:

$$
S_i=\sum_{j\le i}\phi(K_j)V_j^T,\qquad
Z_i=\sum_{j\le i}\phi(K_j),\qquad
V'_i=\frac{\phi(Q_i)^TS_i}{\phi(Q_i)^TZ_i}.
$$

The recurrence is

$$
S_i=S_{i-1}+\phi(K_i)V_i^T,\qquad
Z_i=Z_{i-1}+\phi(K_i),
$$

which gives constant-memory autoregressive decoding.

With layer projections restored,

$$
s_i=s_{i-1}+\phi(x_iW_K)(x_iW_V)^T,\qquad
z_i=z_{i-1}+\phi(x_iW_K),
$$

$$
y_i=f_l\!\left(\frac{\phi(x_iW_Q)^Ts_i}{\phi(x_iW_Q)^Tz_i}+x_i\right).
$$

The causally masked layer is therefore an RNN over time with fixed state `(s,z)`.

## Constant-Memory Gradient

For the causal numerator, absorb `phi` into `Q,K`:

$$
\bar V_i=Q_i^T\sum_{j\le i}K_jV_j^T.
$$

Given `G_i = \nabla_{\bar V_i}L`,

$$
\nabla_{Q_i}L
=
G_i\left(\sum_{j\le i}K_jV_j^T\right)^T,
$$

$$
\nabla_{K_i}L
=
\left(\sum_{j\ge i}Q_jG_j^T\right)V_i,
\qquad
\nabla_{V_i}L
=
\left(\sum_{j\ge i}Q_jG_j^T\right)^TK_i.
$$

`Q` uses a forward cumulative sum. `K` and `V` use the same reverse cumulative sum. This keeps the numerator forward/backward pass linear in `N` without storing every prefix matrix.

## Canonical Implementation

The canonical implementation is `idiap/fast-transformers`. Its essential pieces are:

```python
import torch
from torch.nn import Module

from fast_transformers.causal_product import causal_dot_product
from fast_transformers.feature_maps import elu_feature_map
```

```python
class LinearAttention(Module):
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6):
        super().__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps

    def forward(self, queries, keys, values, attn_mask,
                query_lengths, key_lengths):
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)

        if not attn_mask.all_ones:
            raise RuntimeError("LinearAttention does not support arbitrary attention masks")
        K = K * key_lengths.float_matrix[:, :, None, None]

        KV = torch.einsum("nshd,nshm->nhmd", K, values)
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)
        return V.contiguous()
```

```python
def causal_linear(Q, K, V):
    Q = Q.permute(0, 2, 1, 3).contiguous()
    K = K.permute(0, 2, 1, 3).contiguous()
    V = V.permute(0, 2, 1, 3).contiguous()
    V = causal_dot_product(Q, K, V)
    return V.permute(0, 2, 1, 3).contiguous()


class CausalLinearAttention(Module):
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6):
        super().__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps

    def _make_sizes_compatible(self, Q, K):
        N, L, H, E = Q.shape
        _, S, _, _ = K.shape
        if L == S:
            return Q, K
        if L < S:
            return Q, K[:, :L, :, :]
        return Q, torch.cat([K, K.new_zeros(N, L - S, H, E)], dim=1)

    def forward(self, queries, keys, values, attn_mask,
                query_lengths, key_lengths):
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)

        if not attn_mask.lower_triangular:
            raise RuntimeError("CausalLinearAttention only supports full lower triangular masks")
        K = K * key_lengths.float_matrix[:, :, None, None]
        Q, K = self._make_sizes_compatible(Q, K)

        Z = 1 / (torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + self.eps)
        V = causal_linear(Q, K, values)
        return V * Z[:, :, :, None]
```

```python
class RecurrentLinearAttention(Module):
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6):
        super().__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps

    def forward(self, query, key, value, state=None, memory=None):
        if state is not None and memory is not None:
            raise ValueError("Pass either state or memory, not both")
        if state is None:
            state = memory
        if state is None:
            self.feature_map.new_feature_map(query.device)

        Q = self.feature_map.forward_queries(query)
        K = self.feature_map.forward_keys(key)
        N, H, D = Q.shape
        _, _, M = value.shape

        if state is None:
            S = query.new_zeros((N, H, D, M))
            Z = query.new_zeros((N, H, D))
        else:
            S, Z = state
        if len(S) != N:
            raise ValueError("The batch size changed during iteration")

        if K.grad_fn is not None or value.grad_fn is not None:
            Z = Z + K
            S = S + torch.einsum("nhd,nhm->nhdm", K, value)
        else:
            Z += K
            S += torch.einsum("nhd,nhm->nhdm", K, value)

        denom = 1 / (torch.einsum("nhd,nhd->nh", Q, Z) + self.eps)
        V = torch.einsum("nhd,nhdm,nh->nhm", Q, S, denom)
        return V, [S, Z]
```

The shipped `causal_dot_product` is a custom autograd function backed by CPU/CUDA kernels. Its forward pass accumulates `K_i V_i^T` prefixes; its backward pass uses the forward cumulative matrix for `grad_Q` and the reverse cumulative matrix for `grad_K` and `grad_V`.
