# DeltaNet, distilled

DeltaNet is a linear-attention sequence layer whose fixed-size matrix state is updated by the
**delta rule** (Widrow-Hoff / LMS) instead of an additive outer-product write, together with a
**hardware-efficient chunkwise-parallel training algorithm** that makes the otherwise-sequential
delta recurrence rich in matrix multiplications. The state update multiplies the previous state
by a generalized Householder matrix `I - beta_t k_t k_t^T` and adds a rank-one write; expressed
over a chunk, the per-token writes are obtained in closed form via the WY representation of
Householder products and a triangular matrix inverse (the UT transform), so training runs in
`O(LCd + Ld^2)` FLOPs with `O(L/C)` sequential steps on tensor cores. It gives constant-memory
inference while using an error-correcting write that targets the cross-talk failure of additive
linear attention.

## Problem it solves

A subquadratic, constant-memory-inference replacement for softmax attention that closes the
associative-recall gap to softmax attention, *and* can be trained efficiently on GPUs — i.e.
with a matmul-rich, sequence-parallel algorithm that does not materialize the `d x d` state at
every step.

## Key idea

Read the linear-attention state `S` as a fast-weight associative memory and replace the additive
Hebbian write with an error-correcting one. Treat `S` as a regressor mapping `k_t` to `v_t` and
take one gradient step on `L_t(S) = 1/2 || S k_t - v_t ||^2`:

```
S_t = S_{t-1} - beta_t (S_{t-1} k_t - v_t) k_t^T
    = S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T,     o_t = S_t q_t,
```

with `beta_t = sigma(W_beta x_t) in (0,1)` a learned writing strength. Equivalently: retrieve
`v_t^old = S_{t-1} k_t`, blend `v_t^new = beta_t v_t + (1-beta_t) v_t^old`, and swap it in. The
write is proportional to the prediction error, so it removes the stale association for a
colliding key (content-addressed deallocation) instead of merely accumulating — fixing the
capacity/cross-talk failure of the additive rule `S_t = S_{t-1} + v_t k_t^T`, whose readout
`S k_j = v_j (k_j^T k_j) + sum_{i != j} (k_i^T k_j) v_i` corrupts retrieval once `L` exceeds
the key dim.

## Why it parallelizes

The delta write depends on the running state (`v_t^old = S_{t-1} k_t`), which blocks the
parallel/chunkwise machinery that makes additive linear attention trainable. Two reductions
remove the obstacle.

**Pseudo-value (additive form).** By induction, `S_t = sum_{i<=t} u_i k_i^T` with

```
u_t = beta_t ( v_t - sum_{i<t} u_i (k_i^T k_t) ).
```

So the delta-rule layer is vanilla linear attention with `v_i` replaced by the pseudo-value `u_i`; once `U`
is known, `O = (Q K^T ⊙ M) U`, and the per-token matrix states are not materialized.

**WY representation of the decay.** The product of transition matrices satisfies
`prod_{t=1}^n (I - beta_t k_t k_t^T) = I - sum_{t=1}^n w_t k_t^T` with the same recurrence,
`w_t = beta_t ( k_t - sum_{i<t} w_i (k_i^T k_t) )` (i.e. `u` with `k` in place of `v`).

**Chunkwise form.** With chunks of size `C`, carrying a chunk state `S_[t]` (`d x d`), the
effective intra-chunk write is `U_[t] - W_[t] S_[t]^T`, and (mirroring additive linear
attention with `V` -> `U - W S^T`):

```
S_[t+1] = S_[t] + (U_[t] - W_[t] S_[t]^T)^T K_[t],
O_[t]   = Q_[t] S_[t]^T + (Q_[t] K_[t]^T ⊙ M)(U_[t] - W_[t] S_[t]^T).
```

**UT transform.** The `u`/`w` recurrences are one lower-triangular system. With `B = diag(beta)`
and `L = tril(B K K^T, -1)`, `(I + L) W = B K`, hence

```
T = (I + tril(diag(beta) K K^T, -1))^{-1} diag(beta),     W = T K,    U = T V,
```

and `I + L` is unit lower-triangular, inverted cheaply by forward substitution (all matmuls).
This makes the whole algorithm tensor-core-friendly: `O(LCd + Ld^2)` FLOPs, `O(L/C)` sequential
steps; chunk states are recomputed in the backward pass to save memory. A fully parallel form
exists (`A = (Q K^T ⊙ M) T`, `A_{ij} = k_j^T P_{j+1}^i q_i`) but its `L x L` inverse scales
cubically, so chunking is used for training.

## Design choices and why

- **L2-normalize `q, k`** (vs. L1). The eigenvalues of `I - beta_t k_t k_t^T` are `1`
  (multiplicity `d-1`) and `1 - beta_t ||k_t||^2`. With `||k_t||_2 = 1` and `beta_t in (0,1)`
  the non-unit eigenvalue is `1 - beta_t in [0,1]` — always stable. At `beta_t = 1`,
  `I - k_t k_t^T` is an orthogonal projection: it erases exactly the `k_t` direction and keeps
  the other `d-1` subspaces — *targeted* forgetting. L1 normalization bounds the norm, but L2
  makes this projection interpretation exact.
- **SiLU feature map** on `q, k` (then L2-normalize), rather than `elu+1`: positivity is no
  longer needed after dropping the linear-attention normalizer, and SiLU gives a smooth gated
  nonlinearity before projecting directions onto the unit sphere.
- **`beta_t = sigma(W_beta x_t)`**, one scalar per head; negligible parameters
  (`W_beta: d -> num_heads`).
- **Short (depthwise) convolution, kernel 4**, on `q/k/v` after projection: cheap local token
  mixing for shifts and nearby comparisons that pure content-based addressing handles poorly.
- **Output RMSNorm before `o_proj`**: stability for linear-attention layers.
- **Query scaling `d_k^{-1/2}`** before the `Q K^T` products (softmax-attention-style).
- **Chunk size `C` = 64/128** (multiple of 16 for tensor cores): interpolates parallel (`C=L`)
  and recurrent (`C=1`).

## Final form — training (chunkwise) op

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

## Final form — inference (recurrent) op

```python
import torch


def delta_rule_recurrence(q, k, v, beta, S=None):
    """Constant-memory recurrent form (one step at a time), for inference."""
    b, h, L, d_k = q.shape
    q = q * (d_k ** -0.5)
    o = torch.zeros_like(v)
    if S is None:
        S = q.new_zeros(b, h, d_k, v.shape[-1])
    for i in range(L):
        k_i, q_i = k[:, :, i], q[:, :, i]
        v_old = (S * k_i[..., None]).sum(-2)                  # transposed layout: k_i^T S
        u_i = (v[:, :, i] - v_old) * beta[:, :, i, None]      # beta (v - v_old)
        S = S + k_i.unsqueeze(-1) * u_i.unsqueeze(-2)         # S += k_i u_i^T
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', q_i, S)    # transposed layout: q_i^T S
    return o, S
```

The readable chunkwise reference above uses a compact head-first layout. The production FLA layer uses `[B,T,H,D]`, computes the same WY representation in
`chunk_delta_rule`, uses `fused_recurrent_delta_rule` for short/incremental runs, and treats the
old fused-chunk path as deprecated.

## The layer

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

DeltaNet replaces softmax self-attention in a standard pre-norm transformer block
(`x = x + DeltaNet(LN(x))`, `x = x + SwiGLU_FFN(LN(x))`), using no absolute position embeddings
since the recurrence carries order. The same block can also sit inside hybrid stacks that include
sliding-window or occasional global softmax-attention layers.

## Relation to prior methods

- **Additive linear attention** (`S_t = S_{t-1} + v_t k_t^T`) has the same fixed-size
  state/readout but uses the linear-loss / Hebbian write with no error correction; bounded
  capacity, cross-talk.
- **General associative RNN** `S_t = S_{t-1} • M_t + v_t k_t^T`: with `• = ⊙` and structured
  `M_t` this is GLA / RetNet / Mamba / RWKV-6 / HGRN2 / mLSTM (elementwise decay). DeltaNet uses
  matrix multiplication with `M_t = I - beta_t k_t k_t^T`, a structured (diagonal-minus-low-rank,
  `D=I`, `a_t = beta_t k_t`, `b_t = k_t`) transition that models content-dependent interactions
  beyond elementwise gating; the chunkwise algorithm generalizes to the full DPLR
  `M_t = diag(d_t) - a_t b_t^T`.
