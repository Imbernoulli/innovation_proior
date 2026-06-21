Self-attention has become the central primitive of sequence models, but its cost grows quadratically with sequence length. For a sequence of $N$ positions the layer forms the full $N \times N$ matrix $QK^T$ of pairwise similarities, normalizes it with a rowwise softmax, and uses it to mix the value vectors. That is $\mathcal{O}(N^2)$ in both time and memory, and the memory cost is structural rather than incidental, because the entire similarity matrix has to be retained to backpropagate through it. Two problems follow. Training and full-sequence inference are quadratic, so usable context is capped and long-range dependencies get clipped simply because longer sequences do not fit. And, more painfully, autoregressive generation is intrinsically sequential — each token is produced from all the tokens before it — so under softmax attention the per-step cost grows with the prefix, since the new query must be compared against every past key. Caching past keys and values avoids recomputing old projections but does nothing about this growth: generating a sequence of length $N$ still costs $\mathcal{O}(N^2)$ and the per-step work is never constant. Producing an image one pixel at a time, where a single CIFAR-10 image is over three thousand positions, becomes prohibitively slow.

The existing escape routes all fall short of what we actually want. Sparse attention (Child et al., 2019) replaces the dense matrix with hand-designed sparse factorizations, reaching $\mathcal{O}(N\sqrt N)$, but it reduces the cost without linearizing it and still has no constant-per-step decoding. Reformer (Kitaev et al., 2020) hashes queries and keys into shared buckets for $\mathcal{O}(N\log N)$, but to put them in the same buckets it forces keys to equal queries, injects noise through the hashing, and again only accelerates training, not per-step inference. Context-extension methods like Transformer-XL and adaptive-span keep the same asymptotic complexity. Recurrent networks have exactly the inference profile we want — constant per-step cost and bounded memory, since they fold the past into a fixed-size hidden state — but they were displaced by attention on quality and are sequential to train. The goal is therefore an attention mechanism that is $\mathcal{O}(N)$ in time and memory, that supports causal masking at linear cost so it can be trained autoregressively in parallel, and that decodes at constant per-step cost regardless of how much has already been generated.

I propose Linear Attention. The starting move is to stop treating softmax as the definition of attention and read it instead as a normalized kernel smoother, $V'_i = \big(\sum_j \operatorname{sim}(Q_i,K_j)V_j\big) / \big(\sum_j \operatorname{sim}(Q_i,K_j)\big)$, which recovers softmax exactly when $\operatorname{sim}(q,k)=\exp(q^Tk/\sqrt D)$. This form only requires that the similarity be non-negative with a nonzero denominator, so the exponential is not sacred; I am free to pick any non-negative similarity that gives better algebra. The algebra I want is factorization. The exponential binds $q$ and $k$ together inside a nonlinearity, so every pair $(i,j)$ is an irreducible computation. Instead I choose a similarity that factors through a feature map, $\operatorname{sim}(q,k)=\phi(q)^T\phi(k)$ with non-negative coordinates, and substitute it into the smoother:
$$
V'_i =
\frac{\sum_{j=1}^N \phi(Q_i)^T\phi(K_j)V_j}
     {\sum_{j=1}^N \phi(Q_i)^T\phi(K_j)} .
$$
Now $\phi(Q_i)$ does not depend on the summation index $j$, so I can pull it outside both sums:
$$
V'_i =
\frac{\phi(Q_i)^T\left(\sum_{j=1}^N \phi(K_j)V_j^T\right)}
     {\phi(Q_i)^T\left(\sum_{j=1}^N \phi(K_j)\right)} .
$$
This is the whole idea. The key-value sum $\sum_j \phi(K_j)V_j^T$ is a $C \times M$ matrix and the key sum $\sum_j \phi(K_j)$ is a $C$-vector; neither depends on the query, so I compute each once and then contract every query against the two summaries. It is nothing more than choosing the parenthesization $\phi(Q)(\phi(K)^TV)$ over $(\phi(Q)\phi(K)^T)V$, but that choice never materializes the $N \times N$ matrix and brings the work down to $\mathcal{O}(NCM)$, linear in sequence length.

The feature map is the next decision, and it must be made honestly. The exponential kernel's exact feature map is infinite-dimensional, so I am not making softmax cheap — I am defining a different, finite-dimensional attention kernel. A degree-2 polynomial kernel has a finite map, but its dimension scales like $D^2$, attractive only when $N \gg D^2$; for a cheap general-purpose layer I want $C = D$. So I need an elementwise positive map of dimension $D$. ReLU is non-negative but kills the gradient on negative inputs; ELU bottoms out at $-1$ and keeps a live derivative on the negative side, so shifting it up by one gives a clean choice,
$$
\phi(x)=\operatorname{elu}(x)+1 ,
$$
which is non-negative everywhere, leaves the feature dimension at $D$, and never starves the gradient. That yields a valid finite-dimensional kernel attention with $\mathcal{O}(NDM)$ attention work.

Causal masking is where the construction has to be done carefully, because dropping a triangular mask onto an explicit score matrix would resurrect exactly the matrix I removed. So I write the causal output directly, restricting each query to its own prefix:
$$
V'_i =
\frac{\phi(Q_i)^T\sum_{j=1}^i \phi(K_j)V_j^T}
     {\phi(Q_i)^T\sum_{j=1}^i \phi(K_j)} .
$$
The two summaries now depend on $i$, but they are prefix sums. Defining $S_i=\sum_{j\le i}\phi(K_j)V_j^T$ and $Z_i=\sum_{j\le i}\phi(K_j)$, the output is $V'_i = \phi(Q_i)^T S_i / \phi(Q_i)^T Z_i$ and the prefixes obey the simple recurrence
$$
S_i=S_{i-1}+\phi(K_i)V_i^T,\qquad
Z_i=Z_{i-1}+\phi(K_i).
$$
This is precisely the fixed-size state I was after. With the projections restored, for a layer input $x_i$,
$$
s_i=s_{i-1}+\phi(x_iW_K)(x_iW_V)^T,\qquad
z_i=z_{i-1}+\phi(x_iW_K),
$$
$$
y_i=f_l\!\left(\frac{\phi(x_iW_Q)^Ts_i}{\phi(x_iW_Q)^Tz_i}+x_i\right),
$$
so the causally masked layer is literally a recurrent network over time with hidden state $(s,z)$. The recurrence is not an added architectural ornament; it is what causally masked attention becomes once the kernel factorizes. At generation time I carry forward only $(s,z)$: each new token applies one rank-one update and one vector update, then reads out its value by two contractions, and the state size never grows with the sequence — constant time and constant memory per step.

There is one remaining trap in training. If I compute every prefix state $S_i$ naively and let autodiff save all of them, I have thrown away the memory gain, because each $S_i$ is a $C \times M$ matrix and storing one per position is the very thing I wanted to avoid. The backward pass has to be a scan as well. Absorbing $\phi$ into $Q,K$ and writing the causal numerator $\bar V_i = Q_i^T\sum_{j\le i}K_jV_j^T$, with $G_i=\nabla_{\bar V_i}L$, the three gradients fall out of how each variable participates. The query $Q_l$ enters only its own output, so its gradient is a forward cumulative sum, $\nabla_{Q_i}L = G_i\big(\sum_{j\le i}K_jV_j^T\big)^T$, reusing the same forward prefix matrix as the numerator. The key $K_l$ and value $V_l$ are written into every later prefix and so contribute to all outputs $i\ge l$, which makes their gradients reverse cumulative sums sharing one matrix $R_i=\sum_{j\ge i}Q_jG_j^T$: $\nabla_{K_i}L = R_iV_i$ and $\nabla_{V_i}L = R_i^TK_i$. The cases line up cleanly because the numerator is additive and multilinear, the only nonlinearity (the denominator) is handled by the final division, and there are no stray signs or constants. With the forward cumulative scan for $Q$ and the shared reverse cumulative scan for $K$ and $V$, causal training stays linear in $N$ and never stores the full list of prefix matrices.

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
