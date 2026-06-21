The core problem is the cost of softmax attention in autoregressive language modeling. Every layer forms an L×L score matrix, so compute and memory grow quadratically with sequence length, and at inference the key-value cache grows without bound. Linear attention removes that quadratic term by replacing the exponential softmax kernel with a plain dot-product kernel, which lets the causal sum collapse into a recurrent matrix state. That gives O(1)-per-step inference and subquadratic training, but the classic update S_t = S_{t-1} + k_t^T v_t only ever accumulates outer products; it never forgets. Old, irrelevant content lingers in the fixed-size state and dilutes the present, which is why plain linear attention underperforms softmax on language modeling quality. RetNet improves things by multiplying the state with a single global scalar γ before each update, but that decay is data-independent: one forgetting rate for every token, channel, and context, so it cannot look at the content and decide what to keep. A fully data-dependent matrix gate would fix that, yet the fine-grained gated linear-attention and selective state-space work shows that full-rank, input-dependent transitions destroy the matmul structure and force slow, memory-bound training.

I propose Gated Linear Attention, or GLA. It adds a per-key-channel, data-dependent forget gate to linear attention while preserving the tensor-core-friendly chunkwise form. The recurrence is S_t = Diag(α_t) S_{t-1} + k_t^T v_t and the readout is o_t = (q_t / sqrt(d_k)) S_t, where α_t ∈ (0,1)^{d_k} is computed from the current input x_t alone. Because the gate depends only on the current input and not on the previous state, the recurrence stays linear and parallelizable. The diagonal shape is the key structural choice: writing the gate as an outer product α_t^T 1 means that when the recurrence is unrolled, the per-step gates telescope into a single cumulative product b_t = ∏_{j≤t} α_j. The layer then reduces to plain linear attention on preconditioned tensors, Q̃ = Q ⊙ B and K̃ = K / B with B = stack(b_t), so the chunkwise matmul training form survives. That is what a full-rank data-dependent transition cannot offer.

The cumulative product b_t underflows over long sequences, which would make K / B explode, so the gate is stored and applied in log space: log α_t = (1/τ) log σ(x_t W_α^1 W_α^2 + b_α), with temperature τ = 16 to bias the initialized gate toward 1, giving a slow-forgetting prior and keeping the log-decay small. Computing scores as Σ_k Q_ik K_jk exp(log B_ik − log B_jk) is stable but not a pure matmul, so a two-level chunking strategy isolates that log-space work to small diagonal sub-blocks inside each chunk. All off-diagonal sub-block interactions and the inter-chunk state recurrence stay as half-precision tensor-core matmuls. I/O is handled with the same tiling and recomputation ideas as FlashAttention: a materialization pass can store chunk states for sequence-level parallelism, and the states are recomputed in the backward pass instead of being kept in HBM. The gate gradient has a closed form, d log b_t = q_t ⊙ dq_t − k_t ⊙ dk_t, so per-step states never need to be materialized for the backward.

For capacity I keep the value dimension at full width, d_v = d, and set d_k = d/2, so the recurrent state is a wide d/2 × d matrix. With full-rank W_Q, W_K, W_V, W_O and a low-rank rank-16 gate projection, one GLA layer lands near the same 4d² parameter budget as softmax attention. The gate is per-head, each head output is RMSNorm-normalized before concatenation, and a Swish output gate multiplies the merged output before the final projection. Because the cumulative decay already encodes a data-dependent relative position bias, no absolute positional embeddings are needed; the mixer plugs into the standard pre-norm Transformer block in place of softmax attention.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def naive_recurrent_gla(q, k, v, gk, scale=None):
    """Reference recurrence the chunkwise kernel parallelizes.
    q, k, gk: (B, T, H, d_k);  v: (B, T, H, d_v).  gk = log forget gate (<= 0)."""
    q, k, v, gk = (x.transpose(1, 2).float() for x in (q, k, v, gk))
    B, H, T, d_k = q.shape
    d_v = v.shape[-1]
    scale = d_k ** -0.5 if scale is None else scale
    h = q.new_zeros(B, H, d_k, d_v)
    o = torch.zeros_like(v)
    for i in range(T):
        q_i = q[:, :, i] * scale
        kv_i = k[:, :, i][..., None] * v[:, :, i][..., None, :]
        h = h * gk[:, :, i].exp()[..., None] + kv_i
        o[:, :, i] = (q_i[..., None] * h).sum(-2)
    return o.transpose(1, 2)


class GatedLinearAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, expand_k=0.5, expand_v=1.0,
                 gate_low_rank_dim=16, gate_logit_normalizer=16,
                 use_output_gate=True, norm_eps=1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.gate_logit_normalizer = gate_logit_normalizer
        self.use_output_gate = use_output_gate
        self.use_pos_emb = False

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, self.key_dim, bias=True),
        )
        self.g_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, x):
        q = rearrange(self.q_proj(x), '... (h d) -> ... h d', d=self.head_k_dim)
        k = rearrange(self.k_proj(x), '... (h d) -> ... h d', d=self.head_k_dim)
        v = rearrange(self.v_proj(x), '... (h d) -> ... h d', d=self.head_v_dim)
        gk = rearrange(self.gk_proj(x), '... (h d) -> ... h d', d=self.head_k_dim)
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        o = naive_recurrent_gla(q, k, v, gk)
        o = self.g_norm(o)
        o = rearrange(o, '... h d -> ... (h d)')
        if self.use_output_gate:
            o = o * F.silu(self.g_proj(x))
        return self.o_proj(o)
```

In production training the naive recurrent loop is replaced by the hardware-efficient chunkwise kernel, which computes exactly the same outputs but routes the bulk of the work through tensor cores, keeps only the small diagonal sub-blocks in full-precision log space, and uses recomputation to stay memory-efficient. The result is a sequence mixer that scales linearly at inference, trains subquadratically, and closes the quality gap with softmax attention by finally giving linear attention a data-dependent forget gate.
