# Linear Attention (Linear Transformers)

## Problem

Softmax self-attention, $V' = \text{softmax}(QK^T/\sqrt{D})\,V$, forms an
$N\times N$ similarity matrix and is therefore $\mathcal{O}(N^2)$ in both time
and memory (the matrix must also be stored for the backward pass). Worse, its
autoregressive generation has no constant-cost step: emitting token $i$ requires
attending over all $i$ prior tokens, so a full sequence costs $\mathcal{O}(N^2)$
and per-step cost grows with context. Linear attention removes the $N^2$ entirely
and turns causal generation into an RNN with constant cost per step.

## Key idea

Write attention as a kernel smoother,
$$ V'_i = \frac{\sum_{j} \text{sim}(Q_i,K_j)\,V_j}{\sum_{j} \text{sim}(Q_i,K_j)}, $$
which is valid for any **non-negative** similarity (softmax is the special case
$\text{sim}(q,k)=\exp(q^Tk/\sqrt{D})$). Choose a similarity that **factors** as a
dot product of feature maps, $\text{sim}(q,k)=\phi(q)^T\phi(k)$. Then the numerator
$\sum_j \phi(Q_i)^T\phi(K_j)V_j = \phi(Q_i)^T\sum_j \phi(K_j)V_j^T$, and by
**associativity** the key-side sums no longer depend on the query:
$$ V'_i = \frac{\phi(Q_i)^T \sum_{j} \phi(K_j) V_j^T}{\phi(Q_i)^T \sum_{j} \phi(K_j)}
\;\;\Longleftrightarrow\;\; (\phi(Q)\phi(K)^T)V = \phi(Q)(\phi(K)^T V). $$
The two sums are computed once and reused for every query → $\mathcal{O}(N)$ time
and memory, no $N\times N$ matrix.

**Feature map.** Use $\phi(x) = \text{elu}(x) + 1$: non-negative (so attention is
well-defined), $\mathcal{O}(D)$ cheap, keeps the feature dimension $C=D$, and —
unlike $\text{relu}(x)+1$ — has a nonzero gradient for $x<0$. (Exact softmax cannot
be linearized: its feature map is infinite-dimensional. A degree-2 polynomial
kernel has an exact finite map at $\mathcal{O}(ND^2M)$, useful when $N>D^2$.)

**Causal masking.** With the mask, sums run to $i$. Define prefix-sum states
$$ S_i = \sum_{j\le i}\phi(K_j)V_j^T, \qquad Z_i = \sum_{j\le i}\phi(K_j), \qquad
V'_i = \frac{\phi(Q_i)^T S_i}{\phi(Q_i)^T Z_i}, $$
with the recurrence $S_i = S_{i-1} + \phi(K_i)V_i^T$, $Z_i = Z_{i-1} + \phi(K_i)$.
One linear pass; constant work per step.

**Transformers are RNNs.** A causally-masked layer is exactly a recurrent network
with fixed-size hidden state $(s,z)$ (recurrence over *time*, not depth):
$$
s_0=0,\; z_0=0,\quad
s_i = s_{i-1} + \phi(x_iW_K)(x_iW_V)^T,\quad
z_i = z_{i-1} + \phi(x_iW_K),\quad
y_i = f_l\!\left(\frac{\phi(x_iW_Q)^T s_i}{\phi(x_iW_Q)^T z_i} + x_i\right).
$$
Train in parallel (all prefix sums at once, like a transformer); generate
sequentially at $\mathcal{O}(1)$ per step and constant memory (like an RNN).

**Constant-memory training.** A naive autograd implementation stores every $S_i$
(memory $\times\max(D,M)$). Instead, derive the numerator gradients as cumulative
sums. With $\bar V_i = Q_i^T\sum_{j\le i}K_jV_j^T$ (absorbing $\phi$ into $Q,K$):
$$
\nabla_{Q_i}\mathcal{L} = \nabla_{\bar V_i}\mathcal{L}\Big(\sum_{j\le i}K_jV_j^T\Big)^{\!T}
\;\;(\text{forward cumsum}),
$$
$$
\nabla_{K_i}\mathcal{L} = \Big(\sum_{j\ge i}Q_j(\nabla_{\bar V_j}\mathcal{L})^T\Big)V_i,
\qquad
\nabla_{V_i}\mathcal{L} = \Big(\sum_{j\ge i}Q_j(\nabla_{\bar V_j}\mathcal{L})^T\Big)^{\!T}K_i
\;\;(\text{reverse cumsum}).
$$
Forward and backward are both $\mathcal{O}(NCM)$ time, $\mathcal{O}(N\max(C,M))$
memory.

## Algorithm (causal numerator)

```
forward(Q, K, V):                            # Q and K are feature-mapped
    S <- 0
    for i = 1..N:
        S <- S + K_i V_i^T
        Vbar_i <- Q_i S
    return Vbar

backward(Q, K, V, G):                       # G = dL/dVbar
    S <- 0
    for i = 1..N:                          # forward cumsum -> grad Q
        S <- S + K_i V_i^T
        grad_Q_i <- G_i S^T
    S <- 0
    for i = N..1:                          # reverse cumsum -> grad K, V
        S <- S + Q_i G_i^T
        grad_V_i    <- S^T K_i
        grad_K_i <- S V_i
    return grad_Q, grad_K, grad_V
```

## Code

```python
import torch
from torch.nn import Module


class ActivationFunctionFeatureMap(Module):
    def __init__(self, query_dimensions, activation_function):
        super().__init__()
        self.query_dimensions = query_dimensions
        self.activation_function = activation_function

    def new_feature_map(self, device):
        return

    def forward_queries(self, x):
        return self.activation_function(x)

    def forward_keys(self, x):
        return self.activation_function(x)


def elu_feature_map(query_dimensions):
    return ActivationFunctionFeatureMap(
        query_dimensions,
        lambda x: torch.nn.functional.elu(x) + 1,
    )


class LinearAttention(Module):
    """Unmasked O(N) attention: form the key/value sums once, reuse per query."""
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
            raise RuntimeError("LinearAttention only supports full masks")
        K = K * key_lengths.float_matrix[:, :, None, None]
        KV = torch.einsum("nshd,nshm->nhmd", K, values)
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)
        return V.contiguous()


class CausalDotProduct(torch.autograd.Function):
    """Causal numerator with cumulative-sum gradients; no stored S_i list."""
    @staticmethod
    def forward(ctx, Q, K, V):
        ctx.save_for_backward(Q, K, V)
        N, H, L, _ = Q.shape
        M = V.shape[-1]
        out = Q.new_zeros((N, H, L, M))
        for n in range(N):
            for h in range(H):
                S = Q.new_zeros((Q.shape[-1], M))
                for i in range(L):
                    S = S + torch.ger(K[n, h, i], V[n, h, i])   # S_i = S_{i-1} + K_i V_i^T
                    out[n, h, i] = S.t().mv(Q[n, h, i])         # Vbar_i = Q_i^T S_i
        return out

    @staticmethod
    def backward(ctx, G):
        Q, K, V = ctx.saved_tensors
        gQ, gK, gV = torch.zeros_like(Q), torch.zeros_like(K), torch.zeros_like(V)
        N, H, L, _ = Q.shape
        for n in range(N):
            for h in range(H):
                S = Q.new_zeros((Q.shape[-1], V.shape[-1]))      # forward cumsum -> grad Q
                for i in range(L):
                    S = S + torch.ger(K[n, h, i], V[n, h, i])
                    gQ[n, h, i] = S.mv(G[n, h, i])
                S = Q.new_zeros((Q.shape[-1], V.shape[-1]))      # reverse cumsum -> grad K, V
                for i in range(L - 1, -1, -1):
                    S = S + torch.ger(Q[n, h, i], G[n, h, i])
                    gV[n, h, i] = S.t().mv(K[n, h, i])
                    gK[n, h, i] = S.mv(V[n, h, i])
        return gQ, gK, gV


def causal_linear(Q, K, V):
    Q = Q.permute(0, 2, 1, 3).contiguous()
    K = K.permute(0, 2, 1, 3).contiguous()
    V = V.permute(0, 2, 1, 3).contiguous()
    V = CausalDotProduct.apply(Q, K, V)
    return V.permute(0, 2, 1, 3).contiguous()


class CausalLinearAttention(Module):
    """Causal O(N) attention via prefix-sum state (no N x N matrix)."""
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
            raise RuntimeError("CausalLinearAttention requires a causal mask")
        K = K * key_lengths.float_matrix[:, :, None, None]
        Q, K = self._make_sizes_compatible(Q, K)
        Z = 1 / (torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + self.eps)
        V = causal_linear(Q, K, values)
        return V * Z[:, :, :, None]


class RecurrentLinearAttention(Module):
    """Generation-time RNN form: carry (S, Z), update per step in O(1)."""
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
        M = value.shape[-1]
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

The inner attention modules above plug into a standard multi-head attention
layer (shared Q/K/V and output projections) in place of softmax attention: use
`LinearAttention` for encoders, `CausalLinearAttention` for autoregressive
training, and `RecurrentLinearAttention` for fast generation.
