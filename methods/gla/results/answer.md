# Gated Linear Attention (GLA), distilled

GLA is a subquadratic sequence-mixing layer: linear attention with a matrix-valued recurrent
state, augmented with a **per-key-channel, data-dependent forget gate**, and trained with a
hardware-efficient chunkwise matmul form (FlashLinearAttention). It gives linear-time inference
with a fixed-size state and subquadratic training, while staying competitive in language-model
quality with strong softmax-attention Transformers.

## Problem it solves

Softmax attention costs `O(L^2)` in compute and memory and grows an unbounded KV cache at
inference. Plain linear attention is cheap (`O(1)`-per-step inference, subquadratic training)
but loses quality because its additive state `S_t = S_{t-1} + k_t^T v_t` can never *forget*.
A global data-independent decay (RetNet) helps but cannot decide from content what to keep. A
fully data-dependent matrix gate is expressive but breaks the matmul/chunkwise structure
(Mamba, fine-grained matrix-gate models) and forces slow, memory-bound training. GLA wants the
expressivity of a data-dependent gate *and* a tensor-core-friendly chunkwise trainer.

## Key idea

Use an **outer-product gate** `G_t = α_t^T 1`, so the recurrence is

```
S_t = (α_t^T 1) ⊙ S_{t-1} + k_t^T v_t = Diag(α_t) S_{t-1} + k_t^T v_t,
o_t = (q_t / sqrt(d_k)) S_t
```

with `α_t ∈ (0,1)^{d_k}` computed from `x_t` alone. Unrolling, the per-step gates telescope
into a cumulative product `b_t = ∏_{j≤t} α_j`, and the layer collapses to plain linear
attention on preconditioned tensors:

```
b_t = ∏_{j≤t} α_j ,   B = stack(b_t)
P = ((Q / sqrt(d_k)) ⊙ B)(K / B)^T ,   O = (P ⊙ M) V          (M = causal mask)
```

so the matmul chunkwise form is preserved where a full-rank data-dependent transition would
force a serial or scan-style state update.

**Numerical stability.** `b_t` is a product of values `< 1`, underflowing for large `t` and
making `K/B` explode. Compute the scores in log space:

```
P_ij = (1/sqrt(d_k)) Σ_k Q_ik K_jk · exp(log B_ik − log B_jk),   i ≥ j
log B_t = Σ_{j≤t} log α_j
```

The exponent is a data-dependent relative-position factor (a learned, content-dependent ALiBi).

## Chunkwise form (training)

Split into `N = L/C` chunks; carry the chunk state `S_{[i]}`. In the formulas below, each `Q`
factor is the scaled query `Q/sqrt(d_k)`. With chunk-relative cumulative
gates `Λ_{iC+j} = b_{iC+j}/b_{iC}` (decay from chunk start), `Γ_{iC+j} = b_{(i+1)C}/b_{iC+j}`
(decay to chunk end), `γ_{i+1} = b_{(i+1)C}/b_{iC}` (whole-chunk decay):

```
S_{[i+1]}      = (γ_{i+1}^T 1) ⊙ S_{[i]} + (K_{[i+1]} ⊙ Γ_{[i+1]})^T V_{[i+1]}
O^inter_{[i+1]} = (Q_{[i+1]} ⊙ Λ_{[i+1]}) S_{[i]}
O^intra_{[i+1]} = (((Q_{[i+1]}⊙Λ_{[i+1]})(K_{[i+1]}/Λ_{[i+1]})^T) ⊙ M) V_{[i+1]}
```

All cumulative products span at most one chunk, so they stay bounded.

## FlashLinearAttention + GLA, hardware notes

- **Two chunk-loop versions.** *Non-materialization*: walk chunks left-to-right, keep `S` in
  SRAM; parallel over batch×heads×head-dim, no sequence-level parallelism (good at large
  batch). *Materialization*: store all `S_{[n]}` in HBM, then compute all chunk outputs in
  parallel (sequence-level parallelism for high occupancy at small batch), +10–20% memory.
- **Recomputation.** Default = materialization + recomputing the states in the backward pass,
  removing the memory cost.
- **Secondary-level chunking.** The log-space intra-chunk scores are not a matmul, so a chunk
  is tiled into sub-chunks: off-diagonal sub-blocks become half-precision tensor-core matmuls
  `P_{[i][j]} = (Q_{[i]}⊙Λ_{[i]})(K_{[j]}⊙Γ_{[j]}⊙ b_{iC}/b_{(j+1)C})^T`; only the diagonal
  sub-blocks use full-precision log space for stability.
- **Memory-efficient gate gradient (closed form, no per-step states).**

  ```
  d log b_t = q_t ⊙ dq_t − k_t ⊙ dk_t ,    d log α_t = Σ_{t ≤ i ≤ L} d log b_i
  ```

  derived from the log-space output via `∂f(a⊙b)/∂log b = a⊙∂f/∂a` and
  `∂f(a/b)/∂log b = −a⊙∂f/∂a`. `dq, dk` come from the same chunkwise kernel as the forward.

## Layer design and defaults

- **Gate parameterization (low rank + temperature).** `α_t = σ(x_t W_α^1 W_α^2 + b_α)^{1/τ}`
  with `W_α^1 ∈ R^{d×16}`, `W_α^2 ∈ R^{16×d_k}`, `τ = 16`. The rank-16 bottleneck keeps the
  gate nearly free in parameters; `^{1/τ}` biases the gate toward 1 (slow forgetting), and in
  log space it is the well-conditioned `log α_t = (1/τ)·logsigmoid(logits)`.
- **Dimensions.** `d_k = d/2`, `d_v = d` (wide state `d_k × d_v` for memory capacity);
  full-rank `W_Q, W_K, W_V, W_O, W_r`. One GLA layer ≈ `4d^2` params, like softmax attention.
- **Query scaling.** `1/sqrt(d_k)` logit scaling on `q`.
- **Multi-head + output path.** Per head `h`: `S_t^h = ((α_t^h)^T 1)⊙S_{t-1}^h + k_t^{hT}v_t^h`,
  `o_t^h = q_t^h S_t^h`. Normalize each head output, concat, then an output gate and projection:
  `r_t = Swish(x_t W_r + b_r)`, `y_t = (r_t ⊙ concat_h Norm(o_t^h)) W_O`.
- **Block.** Pre-norm: `Y = GLA(Norm(X)) + X`; `X' = SwiGLU(Norm(Y)) + X`,
  `SwiGLU(Z) = (Swish(Z W_1) ⊙ Z W_2) W_3`. No absolute position embeddings (the cumulative
  decay encodes relative position).

## Working code

Faithful to the `fla.layers.GatedLinearAttention` implementation: log-space gate
`logsigmoid(low-rank logits) / gate_logit_normalizer`, `expand_k = 0.5` (`d_k = d/2`),
`expand_v = 1.0` (`d_v = d`), per-head RMSNorm + Swish output gate, `chunk` kernel for training
and a `fused_recurrent` kernel for short sequences / decoding. The reference recurrence
(`naive_recurrent_gla`) makes the math explicit; the chunk kernel computes the same outputs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def naive_recurrent_gla(q, k, v, gk, scale=None):
    """Reference recurrence the chunk kernel parallelizes.
    q, k, gk: (B, T, H, d_k);  v: (B, T, H, d_v).  gk = log forget gate (<= 0)."""
    q, k, v, gk = (x.transpose(1, 2).float() for x in (q, k, v, gk))   # -> (B, H, T, .)
    B, H, T, d_k = q.shape
    d_v = v.shape[-1]
    scale = d_k ** -0.5 if scale is None else scale
    h = q.new_zeros(B, H, d_k, d_v)
    o = torch.zeros_like(v)
    for i in range(T):
        q_i = q[:, :, i] * scale
        kv_i = k[:, :, i][..., None] * v[:, :, i][..., None, :]        # outer product k^T v
        h = h * gk[:, :, i].exp()[..., None] + kv_i                    # forget, then add
        o[:, :, i] = (q_i[..., None] * h).sum(-2)                      # q_t S_t
    return o.transpose(1, 2)


class GatedLinearAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, expand_k=0.5, expand_v=1.0,
                 gate_low_rank_dim=16, gate_logit_normalizer=16,
                 use_output_gate=True, norm_eps=1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)          # d_k = d/2
        self.value_dim = int(hidden_size * expand_v)        # d_v = d
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.gate_logit_normalizer = gate_logit_normalizer  # temperature tau
        self.use_output_gate = use_output_gate
        self.use_pos_emb = False                            # decay encodes relative position

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        # low-rank gate projection: d -> 16 -> d_k
        self.gk_proj = nn.Sequential(nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
                                     nn.Linear(gate_low_rank_dim, self.key_dim, bias=True))
        self.g_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)   # per-head output norm
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, x):                                   # x: (B, T, d)
        q = rearrange(self.q_proj(x), '... (h d) -> ... h d', d=self.head_k_dim)
        k = rearrange(self.k_proj(x), '... (h d) -> ... h d', d=self.head_k_dim)
        v = rearrange(self.v_proj(x), '... (h d) -> ... h d', d=self.head_v_dim)
        # log alpha_t = logsigmoid(low-rank logits) / tau  ==  log( sigmoid(.)^{1/tau} ), <= 0
        gk = rearrange(self.gk_proj(x), '... (h d) -> ... h d', d=self.head_k_dim)
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        # training: replace with chunk_gla (chunkwise + secondary tiling + recomputation);
        # short sequences / decoding: fused_recurrent_gla. Same outputs as this reference loop.
        o = naive_recurrent_gla(q, k, v, gk)               # (B, T, H, d_v)

        o = self.g_norm(o)                                 # normalize each head
        o = rearrange(o, '... h d -> ... (h d)')
        if self.use_output_gate:
            o = o * F.silu(self.g_proj(x))                 # Swish output gate
        return self.o_proj(o)
```
