Softmax attention is the dominant sequence mixer, but its cost is quadratic in sequence length and its inference memory grows linearly because every past key and value must be retained in a KV cache. Linear attention removes that bottleneck by re-associating the causal sum into a matrix-valued linear RNN, S_t = S_{t-1} + v_t k_t^T and o_t = S_t q_t, which stores the whole past in a fixed-size state. That gives subquadratic training and constant-memory inference, yet it does not scale to long or recall-heavy sequences: the additive Hebbian write has bounded capacity, so once the sequence length exceeds the key dimension the keys cannot all be orthogonal and retrieval is corrupted by cross-talk from every non-orthogonal stored association. There is also no mechanism to deallocate a stale key-value pair when a colliding key reappears.

The prior rungs on this ladder try to fix forgetting with decay gates. RetNet uses a fixed scalar decay per head, which is stable but data-independent and therefore forgets at the same rate regardless of content. Gated linear attention makes the decay data-dependent and diagonal, improving bulk language modeling but still scaling the whole state coordinate-wise; it cannot remove the specific stored association that conflicts with the new key, so recall-sensitive tasks still suffer. The real problem is the write rule, not only the decay. We need an error-correcting write that looks at what is already stored and updates only what is wrong, while remaining expressible as dense matrix multiplications so modern GPUs can train it efficiently.

The method I propose is DeltaNet. It keeps the fixed-size state and readout of linear attention but replaces the additive write with the delta rule, also known as the Widrow-Hoff LMS update. Treat the state S as a tiny regressor that should map the current key k_t to the current value v_t, and take one gradient step on one half of the squared prediction error. The update is S_t = S_{t-1} - beta_t (S_{t-1} k_t - v_t) k_t^T, which is the same as S_t = S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T, with o_t = S_t q_t. The scalar beta_t = sigmoid(W_beta x_t) in (0,1) is a learned writing strength. Equivalently, retrieve the old value v_t^old = S_{t-1} k_t, blend it with the new value as beta_t v_t + (1 - beta_t) v_t^old, and swap it in. Because the correction scales with the prediction error, a key already represented well produces almost no change, while a colliding key with a stale value is overwritten. With L2-normalized keys the transition matrix has eigenvalues 1 on the orthogonal complement of k_t and 1 - beta_t along k_t, so the recurrence is always stable, and at beta_t = 1 the transition is an orthogonal projection that erases exactly the key direction while leaving the rest of the memory untouched. This is the targeted, content-addressed forgetting that additive linear attention and elementwise gates cannot provide.

The obstacle to training the delta rule at scale is that the value written at step t depends on the running state through v_t^old, so the writes cannot be precomputed and matmul'd the way additive linear attention writes can. DeltaNet removes that obstacle in two steps. First, by induction the state can be written as a sum of outer products S_t = sum_i u_i k_i^T for pseudo-values u_t = beta_t (v_t - sum_{i<t} u_i (k_i^T k_t)). Likewise the product of transition matrices has a compact WY form I - sum_i w_i k_i^T where w_t satisfies the same triangular recurrence with k in place of v. Second, that recurrence is a single unit lower-triangular linear system: with B = diag(beta) and L = tril(B K K^T, -1), we have (I + L) W = B K, so W = (I + L)^{-1} B K and U = (I + L)^{-1} B V. The inverse of the unit lower-triangular I + L is computed by forward substitution, which is a short loop of matrix multiplications inside each chunk. The chunkwise algorithm then mirrors additive linear attention exactly, only with the value matrix V replaced by the corrected pseudo-values U - W S^T. Everything is dense matmul, giving O(L C d + L d^2) FLOPs with O(L/C) sequential steps for chunk size C. For memory, chunk boundary states are recomputed in the backward pass rather than stored.

The architectural details are chosen to keep the recurrence stable and the code simple. Queries and keys pass through a short depthwise causal convolution of kernel size 4 and a SiLU nonlinearity, then are L2-normalized so the transition is a true projection when beta_t is near one. The query is scaled by d_k^{-1/2} before the Q K^T products. Values receive the same short convolution. The writing strength beta_t is one sigmoid scalar per head computed from the input. Before the output projection, a per-head RMSNorm stabilizes the layer. Below is the training kernel exactly as derived: it pre-multiplies beta into k and v, builds the triangular inverse of I + L by forward substitution within each chunk, and carries the d_k by d_v state S from chunk to chunk while emitting the corrected write u_i - w_i S at every step.

```python
import torch
from einops import rearrange


def delta_rule_chunkwise(q, k, v, beta, chunk_size=64):
    """Chunkwise-parallel delta-rule forward.
    q,k,v: [b, h, L, d_k] (q,k already SiLU + L2-normalized); beta: [b, h, L] in (0,1)."""
    b, h, L, d_k = q.shape
    q = q * (d_k ** -0.5)                  # softmax-style scaling
    v_beta = v * beta[..., None]           # V_beta = diag(beta) V
    k_beta = k * beta[..., None]           # K_beta = diag(beta) K
    assert L % chunk_size == 0

    q, k, v_beta, k_beta = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size),
        (q, k, v_beta, k_beta),
    )

    # A = (I + tril(diag(beta) K K^T, -1))^{-1}; beta is folded into K_beta/V_beta.
    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)        # row r, col i: -beta_r k_r^T k_i
    for i in range(1, chunk_size):                                    # invert (I + L) in place
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone()
                                               * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)

    u = attn @ v_beta                    # U = A V_beta = T V
    w = attn @ k_beta                    # W = A K_beta = T K

    S = k.new_zeros(b, h, d_k, v_beta.shape[-1])      # transposed state [d_k, d_v]
    o = torch.zeros_like(v_beta)
    tri1 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    for i in range(L // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        a_i = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri1, 0)     # intra-chunk causal Q K^T
        u_i = u[:, :, i] - w[:, :, i] @ S                            # corrected writes
        o[:, :, i] = q_i @ S + a_i @ u_i                            # inter- + intra-chunk read
        S = S + k_i.transpose(-1, -2) @ u_i                         # transposed state update
    return rearrange(o, 'b h n c d -> b h (n c) d'), S
```

That function is the whole delta-rule mechanism; the surrounding layer only has to supply the projections, the short convolutions above, and the writing-strength scalar, then hand q, k, v, beta to it. In production the L2-normalization of q and k is fused into the kernel rather than called out as a separate step, and the layer dispatches to a fused single-step recurrent form — the bare S_t = S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T recurrence this was derived from — whenever the sequence is short enough (T <= 64) that chunking would only add overhead; otherwise it calls the chunkwise op above:

```python
import torch
import torch.nn as nn
from einops import rearrange
from fla.modules import RMSNorm, ShortConvolution
from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule


class DeltaNet(nn.Module):
    def __init__(self, hidden_size, num_heads, mode="chunk", conv_size=4, norm_eps=1e-5):
        super().__init__()
        assert mode in ["chunk", "fused_recurrent"]
        self.mode = mode
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)        # writing strength
        self.q_conv1d = ShortConvolution(hidden_size=hidden_size, kernel_size=conv_size,
                                         bias=False, activation="silu")
        self.k_conv1d = ShortConvolution(hidden_size=hidden_size, kernel_size=conv_size,
                                         bias=False, activation="silu")
        self.v_conv1d = ShortConvolution(hidden_size=hidden_size, kernel_size=conv_size,
                                         bias=False, activation="silu")
        self.o_norm = RMSNorm(self.head_dim, eps=norm_eps, dtype=torch.float32)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, recurrent_state=None, use_cache=False):
        q, _ = self.q_conv1d(x=self.q_proj(x), cache=None, output_final_state=False)
        k, _ = self.k_conv1d(x=self.k_proj(x), cache=None, output_final_state=False)
        v, _ = self.v_conv1d(x=self.v_proj(x), cache=None, output_final_state=False)
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b t h d', h=self.num_heads), (q, k, v))
        beta = self.b_proj(x).sigmoid()                  # [B,T,H], beta_t in (0,1)
        mode = "fused_recurrent" if x.shape[1] <= 64 else self.mode
        op = fused_recurrent_delta_rule if mode == "fused_recurrent" else chunk_delta_rule
        o, recurrent_state = op(
            q=q, k=k, v=v, beta=beta,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
        )
        o = self.o_norm(o)
        return self.o_proj(rearrange(o, 'b t h d -> b t (h d)')), recurrent_state
```

This layer drops into a pre-norm transformer block in place of softmax self-attention, with no positional embedding needed since the recurrence itself carries order, giving constant-memory inference together with the targeted, content-addressed forgetting that neither additive linear attention nor the elementwise-gated variants can provide.
