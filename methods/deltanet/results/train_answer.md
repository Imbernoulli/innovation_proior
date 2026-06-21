Softmax attention is the dominant sequence mixer, but its cost is quadratic in sequence length and its inference memory grows linearly because every past key and value must be retained in a KV cache. Linear attention removes that bottleneck by re-associating the causal sum into a matrix-valued linear RNN, S_t = S_{t-1} + v_t k_t^T and o_t = S_t q_t, which stores the whole past in a fixed-size state. That gives subquadratic training and constant-memory inference, yet it does not scale to long or recall-heavy sequences: the additive Hebbian write has bounded capacity, so once the sequence length exceeds the key dimension the keys cannot all be orthogonal and retrieval is corrupted by cross-talk from every non-orthogonal stored association. There is also no mechanism to deallocate a stale key-value pair when a colliding key reappears.

The prior rungs on this ladder try to fix forgetting with decay gates. RetNet uses a fixed scalar decay per head, which is stable but data-independent and therefore forgets at the same rate regardless of content. Gated linear attention makes the decay data-dependent and diagonal, improving bulk language modeling but still scaling the whole state coordinate-wise; it cannot remove the specific stored association that conflicts with the new key, so recall-sensitive tasks still suffer. The real problem is the write rule, not only the decay. We need an error-correcting write that looks at what is already stored and updates only what is wrong, while remaining expressible as dense matrix multiplications so modern GPUs can train it efficiently.

The method I propose is DeltaNet. It keeps the fixed-size state and readout of linear attention but replaces the additive write with the delta rule, also known as the Widrow-Hoff LMS update. Treat the state S as a tiny regressor that should map the current key k_t to the current value v_t, and take one gradient step on one half of the squared prediction error. The update is S_t = S_{t-1} - beta_t (S_{t-1} k_t - v_t) k_t^T, which is the same as S_t = S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T, with o_t = S_t q_t. The scalar beta_t = sigmoid(W_beta x_t) in (0,1) is a learned writing strength. Equivalently, retrieve the old value v_t^old = S_{t-1} k_t, blend it with the new value as beta_t v_t + (1 - beta_t) v_t^old, and swap it in. Because the correction scales with the prediction error, a key already represented well produces almost no change, while a colliding key with a stale value is overwritten. With L2-normalized keys the transition matrix has eigenvalues 1 on the orthogonal complement of k_t and 1 - beta_t along k_t, so the recurrence is always stable, and at beta_t = 1 the transition is an orthogonal projection that erases exactly the key direction while leaving the rest of the memory untouched. This is the targeted, content-addressed forgetting that additive linear attention and elementwise gates cannot provide.

The obstacle to training the delta rule at scale is that the value written at step t depends on the running state through v_t^old, so the writes cannot be precomputed and matmul'd the way additive linear attention writes can. DeltaNet removes that obstacle in two steps. First, by induction the state can be written as a sum of outer products S_t = sum_i u_i k_i^T for pseudo-values u_t = beta_t (v_t - sum_{i<t} u_i (k_i^T k_t)). Likewise the product of transition matrices has a compact WY form I - sum_i w_i k_i^T where w_t satisfies the same triangular recurrence with k in place of v. Second, that recurrence is a single unit lower-triangular linear system: with B = diag(beta) and L = tril(B K K^T, -1), we have (I + L) W = B K, so W = (I + L)^{-1} B K and U = (I + L)^{-1} B V. The inverse of the unit lower-triangular I + L is computed by forward substitution, which is a short loop of matrix multiplications inside each chunk. The chunkwise algorithm then mirrors additive linear attention exactly, only with the value matrix V replaced by the corrected pseudo-values U - W S^T. Everything is dense matmul, giving O(L C d + L d^2) FLOPs with O(L/C) sequential steps for chunk size C. For memory, chunk boundary states are recomputed in the backward pass rather than stored.

The architectural details are chosen to keep the recurrence stable and the code simple. Queries and keys pass through a short depthwise causal convolution of kernel size 4 and a SiLU nonlinearity, then are L2-normalized per head so the transition is a true projection when beta_t is near one. The query is scaled by d_k^{-1/2} before the Q K^T products. Values receive the same short convolution. The writing strength beta_t is one scalar per head computed from the input. Before the output projection, a per-head RMSNorm stabilizes the layer. The code below implements a self-contained training-time DeltaNet mixer; it pads the sequence to a multiple of the chunk size, builds the triangular inverse within each chunk, and carries the d_k by d_v state from chunk to chunk.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class DeltaNet(nn.Module):
    """Self-contained training-time DeltaNet token mixer.

    Replaces softmax attention with a delta-rule linear attention layer.
    Input shape: (batch, seq_len, hidden_size).
    Output shape: (batch, seq_len, hidden_size).
    """

    def __init__(self, hidden_size=1024, num_heads=16, chunk_size=64,
                 conv_size=4, norm_eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        self.q_conv = nn.Conv1d(hidden_size, hidden_size, conv_size,
                                groups=hidden_size, bias=False)
        self.k_conv = nn.Conv1d(hidden_size, hidden_size, conv_size,
                                groups=hidden_size, bias=False)
        self.v_conv = nn.Conv1d(hidden_size, hidden_size, conv_size,
                                groups=hidden_size, bias=False)

        self.o_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _short_conv(self, x, conv):
        # x: (B, T, C) -> causal depthwise conv -> (B, T, C)
        B, T, C = x.shape
        x = x.transpose(1, 2)                       # (B, C, T)
        x = F.pad(x, (conv.kernel_size[0] - 1, 0))
        x = conv(x)[..., :T]
        return x.transpose(1, 2)                    # (B, T, C)

    def forward(self, x):
        B, T, _ = x.shape
        pad = (self.chunk_size - T % self.chunk_size) % self.chunk_size
        if pad:
            x = torch.cat([x, x.new_zeros(B, pad, x.shape[-1])], dim=1)
        Tp = x.shape[1]

        q = F.silu(self._short_conv(self.q_proj(x), self.q_conv))
        k = F.silu(self._short_conv(self.k_proj(x), self.k_conv))
        v = F.silu(self._short_conv(self.v_proj(x), self.v_conv))

        q = q.view(B, Tp, self.num_heads, self.head_dim)
        k = k.view(B, Tp, self.num_heads, self.head_dim)
        v = v.view(B, Tp, self.num_heads, self.head_dim)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        q = q.transpose(1, 2)      # (B, H, Tp, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        beta = self.b_proj(x).sigmoid().transpose(1, 2)  # (B, H, Tp)

        o, _ = delta_rule_chunkwise(q, k, v, beta, self.chunk_size)

        o = o.transpose(1, 2)      # (B, Tp, H, D)
        o = self.o_norm(o)
        o = o.reshape(B, Tp, self.hidden_size)

        if pad:
            o = o[:, :T]
        return self.o_proj(o)


def delta_rule_chunkwise(q, k, v, beta, chunk_size=64):
    """Chunkwise-parallel delta-rule forward pass.

    q, k, v: (B, H, L, D); beta: (B, H, L)
    Returns output (B, H, L, D) and the final state.
    """
    B, H, L, D = q.shape
    q = q * (D ** -0.5)
    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]
    assert L % chunk_size == 0
    n = L // chunk_size

    q = q.view(B, H, n, chunk_size, D)
    k = k.view(B, H, n, chunk_size, D)
    v_beta = v_beta.view(B, H, n, chunk_size, D)
    k_beta = k_beta.view(B, H, n, chunk_size, D)

    # Build A = (I + L)^{-1}, where L = tril(diag(beta) K K^T, -1).
    # beta is already folded into k_beta, so -L = -(k_beta @ k^T).
    triu_mask = torch.triu(torch.ones(chunk_size, chunk_size,
                                      device=q.device, dtype=torch.bool), 0)
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(triu_mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (
            attn[..., i, :, None].clone() * attn[..., :, :i].clone()
        ).sum(-2)
    A = attn + torch.eye(chunk_size, device=q.device, dtype=q.dtype)

    u = A @ v_beta          # pseudo-values U = A V_beta
    w = A @ k_beta          # pseudo-keys   W = A K_beta

    S = q.new_zeros(B, H, D, D)
    o = torch.zeros_like(v_beta)
    causal_mask = torch.triu(torch.ones(chunk_size, chunk_size,
                                        device=q.device, dtype=torch.bool), 1)

    for i in range(n):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        a = (q_i @ k_i.transpose(-1, -2)).masked_fill(causal_mask, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S
        o[:, :, i] = q_i @ S + a @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    return o.view(B, H, L, D), S
```
