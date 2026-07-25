I propose the canonical method name "RetNet," short for Retentive Network, and identify its central operator as multi-scale retention. The problem it addresses is the familiar impossible triangle of autoregressive language modeling: we want training-time parallelism like Transformers, constant-time-and-memory inference like RNNs, and quality that remains competitive with standard attention. Standard softmax attention achieves the first and third of these properties but fails the second, because every decode step must compare the current query with every previous key, producing a key-value cache that grows linearly with sequence length and latency that grows in the same way. Recurrent networks achieve constant inference but forfeit parallel training. Efficient attention variants and state-space models recover one missing corner while giving up another. RetNet is designed to sit in all three corners at once by replacing attention with a single operator that has three mathematically equivalent evaluation modes.

The retention operator is derived from a linear recurrence with a state matrix. Start with S_n = A S_{n-1} + k_n^T v_n and read out o_n = q_n S_n, where k_n and q_n are content projections of the input. Unrolling the recurrence gives o_n = sum_{m<=n} q_n A^{n-m} k_m^T v_m. This expression already looks like causal attention, except the relative-distance weight is carried by a matrix power rather than a softmax score. Diagonalizing A as a rotation-and-decay per dimension and absorbing the change-of-basis matrices into the query and key projections reduces A^{n-m} to a scalar decay gamma^{n-m} multiplied by a rotary relative phase e^{i(n-m)theta}. The decay enforces causality and provides a finite memory horizon, while the rotation supplies the relative position encoding without requiring an additive positional embedding.

For training, the operator is evaluated in a fully parallel form. Let Q, K, V be the usual projections after applying the rotary encoding, and let D be the lower-triangular matrix whose entry (n,m) equals gamma^{n-m} when n is at least m and zero otherwise. Then the retention output is (QK^T circle D) V. This is the same shape as a standard attention forward, two matrix multiplications and an elementwise mask, but the softmax is gone and is replaced by explicit distance-based decay. Because the entire expression is a product of per-position terms, it can be computed over the whole sequence in one GPU-friendly call.

For inference, the same operator is evaluated as a fixed-size recurrence. The state S is updated by S_n = gamma S_{n-1} + K_n^T V_n, and the output at step n is Q_n S_n. The state has constant size d_k by d_v, independent of the generated length, so each decode step performs constant work and stores constant memory. The equivalence with the parallel form follows by unrolling: S_n equals sum_{m<=n} gamma^{n-m} K_m^T V_m, and multiplying by Q_n gives exactly row n of the parallel expression because the causal mask D is the same statement as only past positions being included in the state.

For long-sequence training, a chunkwise form splits the difference. The sequence is divided into fixed-size blocks. Within each block the parallel form is used, preserving matrix-multiplication parallelism; across blocks the recurrence is used, keeping memory linear in sequence length rather than quadratic. The bookkeeping is subtle but exact: when folding a block into the recurrent state, each key is pre-weighted by the decay from its local position to the block boundary, and when a later block reads that state it scales by the decay from the boundary to the query position. The two exponents add up to the true relative distance, so the chunkwise form computes exactly the same retention map as the parallel form.

A single decay rate would restrict the model to one memory horizon, so RetNet uses multiple heads with different gamma values, ranging from quickly forgetting to almost constant. This multi-scale retention lets some heads track local syntactic dependencies while others maintain long-range discourse. Different gamma values produce outputs with different scales, so per-head normalization, implemented as RMSNorm on each head separately, balances their variances. Deleting softmax also removes its nonlinearity, so a content-dependent swish output gate is applied after the head mixing to restore nonlinear, data-dependent gating.

In practice the retention layer is inserted into a Transformer-style pre-norm residual block: layer normalization, multi-scale retention, residual connection, layer normalization, feed-forward network, residual connection. Position embeddings can be dropped because the rotary phase already encodes relative position. The value dimension is often widened to give the recurrent state more capacity, and the feed-forward intermediate dimension is adjusted to keep parameter counts comparable to a Transformer of the same width and depth.

The code below is the reference implementation of multi-scale retention, and it is built exactly around the three faces described above. An RMSNorm module supplies the per-head normalization, kept scale-invariant by never subtracting a mean. A rotate_every_two and theta_shift pair implement the RoPE-style rotation, applied identically to queries and keys so that the rotated dot product depends only on relative position. RetNetRelPos precomputes everything that is shared across heads and does not depend on content: the sin/cos rotation angles, the per-head decay rates gamma_h = 1 - 2^{-5-h}, the causal decay-and-mask matrix D for the parallel path (row-normalized by its square-rooted row sum for numerical stability), the four-way (inner mask, cross-chunk decay, query-side decay, value-side decay) bundle for the chunkwise path, and the bare per-head decay for the recurrent path. MultiScaleRetention then holds the actual projections, q_proj, k_proj, v_proj, g_proj, out_proj, plus the per-head RMSNorm group_norm, and its forward method dispatches to one of three private methods depending on what is passed in: parallel_forward multiplies the rotated Q and K, applies the decay mask, clamps the row-absolute-sum for stability, and multiplies by V; recurrent_forward keeps a running key-value state together with a running scale factor in an incremental_state dictionary, updating both every step so that decode is O(1); chunk_recurrent_forward runs the inner-chunk term as a masked parallel form and the cross-chunk term as a state carried between chunks, then reconciles the two by dividing through by whichever of their two internal scale trackers is larger. After whichever path runs, the output is normalized per head and multiplied by a swish-gated projection of the input before the final output projection.

```python
import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x if self.weight is None else x * self.weight


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class RetNetRelPos(nn.Module):
    def __init__(self, embed_dim, num_heads, chunk_size=512):
        super().__init__()
        angle = 1.0 / (
            10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2)
        )
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(
            1 - 2 ** (-5 - torch.arange(num_heads, dtype=torch.float))
        )
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.recurrent_chunk_size = chunk_size

    def forward(self, slen, activate_recurrent=False, chunkwise_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            return (sin, cos), self.decay.exp()

        index = torch.arange(slen).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :])
        cos = torch.cos(index[:, None] * self.angle[None, :])

        if chunkwise_recurrent:
            b = self.recurrent_chunk_size
            block_index = torch.arange(b).to(self.decay)
            tri = torch.tril(torch.ones(b, b).to(self.decay))
            raw = torch.masked_fill(
                block_index[:, None] - block_index[None, :],
                ~tri.bool(),
                float("inf"),
            )
            raw = torch.nan_to_num(torch.exp(raw * self.decay[:, None, None]))

            value_inner_decay = raw[:, -1] / raw[:, -1].sum(dim=-1, keepdim=True)
            value_inner_decay = value_inner_decay.unsqueeze(-1)
            scale = raw.sum(dim=-1, keepdim=True).sqrt()
            inner_mask = raw / scale

            cross_decay = torch.exp(self.decay * b)[:, None, None]
            query_inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            query_inner_decay = query_inner_decay[:, :, None] / (
                scale / raw[:, -1].sum(dim=-1)[:, None, None]
            )
            return (sin, cos), (
                inner_mask,
                cross_decay,
                query_inner_decay,
                value_inner_decay,
            )

        tri = torch.tril(torch.ones(slen, slen).to(self.decay))
        raw = torch.masked_fill(index[:, None] - index[None, :], ~tri.bool(), float("inf"))
        mask = torch.nan_to_num(torch.exp(raw * self.decay[:, None, None]))
        mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
        return (sin, cos), mask


class MultiScaleRetention(nn.Module):
    def __init__(self, embed_dim, value_dim, num_heads, gate_fn="swish", layernorm_eps=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = value_dim // num_heads
        self.key_dim = embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.gate_fn = F.silu if gate_fn == "swish" else F.gelu

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, value_dim, bias=False)
        self.g_proj = nn.Linear(embed_dim, value_dim, bias=False)
        self.out_proj = nn.Linear(value_dim, embed_dim, bias=False)
        self.group_norm = RMSNorm(
            self.head_dim, eps=layernorm_eps, elementwise_affine=False
        )

    def parallel_forward(self, qr, kr, v, mask):
        bsz, tgt_len, _ = v.size()
        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        qk = (qr @ kr.transpose(-1, -2)) * mask
        qk = qk / qk.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1, max=5e4)
        return (qk @ vr).transpose(1, 2)

    def recurrent_forward(self, qr, kr, v, decay, incremental_state):
        bsz = v.size(0)
        v = v.view(bsz, self.num_heads, self.head_dim, 1)
        kv = kr * v
        if "prev_key_value" in incremental_state:
            prev_kv = incremental_state["prev_key_value"]
            prev_scale = incremental_state["scale"]
            scale = prev_scale * decay + 1
            old = prev_kv * (prev_scale.sqrt() * decay / scale.sqrt()).view(
                self.num_heads, 1, 1
            )
            new = kv / scale.sqrt().view(self.num_heads, 1, 1)
            kv = old + new
        else:
            scale = torch.ones_like(decay)
        incremental_state["prev_key_value"] = kv
        incremental_state["scale"] = scale
        return torch.sum(qr * kv, dim=3)

    def chunk_recurrent_forward(self, qr, kr, v, inner_mask):
        mask, cross_decay, query_inner_decay, value_inner_decay = inner_mask
        bsz, tgt_len, _ = v.size()
        chunk_len = mask.size(1)
        assert tgt_len % chunk_len == 0
        num_chunks = tgt_len // chunk_len

        qr = qr.view(
            bsz, self.num_heads, num_chunks, chunk_len, self.key_dim
        ).transpose(1, 2)
        kr = kr.view(
            bsz, self.num_heads, num_chunks, chunk_len, self.key_dim
        ).transpose(1, 2)
        v = v.view(
            bsz, num_chunks, chunk_len, self.num_heads, self.head_dim
        ).transpose(2, 3)

        kr_t = kr.transpose(-1, -2)
        qk = (qr @ kr_t) * mask
        inner_scale = qk.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
        qk = qk / inner_scale
        inner_output = qk @ v

        kv = kr_t @ (v * value_inner_decay)
        kv_recurrent, cross_scale = [], []
        kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
        kv_scale = torch.ones(bsz, self.num_heads, 1, 1).to(v)
        for i in range(num_chunks):
            kv_recurrent.append(kv_state / kv_scale)
            cross_scale.append(kv_scale)
            kv_state = kv_state * cross_decay + kv[:, i]
            kv_scale = (
                kv_state.detach()
                .abs()
                .sum(dim=-2, keepdim=True)
                .max(dim=-1, keepdim=True)
                .values
                .clamp(min=1)
            )
        kv_recurrent = torch.stack(kv_recurrent, dim=1)
        cross_scale = torch.stack(cross_scale, dim=1)

        all_scale = torch.maximum(inner_scale, cross_scale)
        inner_output = inner_output / (all_scale / inner_scale)
        cross_output = ((qr * query_inner_decay) @ kv_recurrent) / (
            all_scale / cross_scale
        )
        return (inner_output + cross_output).transpose(2, 3)

    def forward(self, x, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        bsz, tgt_len, _ = x.size()
        (sin, cos), inner_mask = rel_pos
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        k = k * self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        if incremental_state is not None:
            output = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state)
        elif chunkwise_recurrent:
            output = self.chunk_recurrent_forward(qr, kr, v, inner_mask)
        else:
            output = self.parallel_forward(qr, kr, v, inner_mask)

        output = self.group_norm(output).reshape(
            bsz, tgt_len, self.head_dim * self.num_heads
        )
        return self.out_proj(self.gate_fn(g) * output)
```
