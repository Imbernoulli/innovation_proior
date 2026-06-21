The self-attention layer computes O = softmax(QK^T)V, but the standard implementation materializes the score matrix S = QK^T and the probability matrix P = softmax(S), both N x N, in high-bandwidth memory. Because softmax, masking and dropout are memory-bound operations that do little arithmetic per byte moved, the wall-clock time is dominated by shuttling these N x N matrices across the slow HBM bus, not by the two dense matmuls. Approximate attention methods reduce FLOPs by sparsifying or low-rank approximating the score matrix, but they change the function and, more importantly, often fail to speed up wall-clock time because FLOPs are not the bottleneck on a memory-bound kernel. The right target is therefore to compute exactly the same attention map while eliminating the N x N traffic.

Two obstacles stand in the way. First, softmax couples an entire row: the denominator sums over all N keys, so it seems the whole row of scores is needed before normalization. Second, the backward pass conventionally needs S and P again, which would force storing them. The solution addresses both by making attention IO-aware: tile Q, K and V into SRAM-sized blocks, combine per-block softmax partials with a running-max recurrence, and recompute the score and probability matrices on-chip during backprop rather than materializing them in HBM.

The method I propose is FlashAttention. It is an exact, fused GPU attention kernel that never writes an N x N object to HBM. The forward pass splits Q into row-blocks and K, V into row-blocks sized to fit in on-chip SRAM. For each query block, one kernel program loads its query rows once, then loops over key/value blocks. For each block it computes the B_r x B_c score tile on-chip, finds the block-local row max and normalizer, and folds them into running per-row state (m, l) using the online-softmax identity: when a new block raises the running max from m to m_new, both the normalizer and the unnormalized output accumulator are rescaled by e^{m - m_new} before the new block's contribution is added. This identity is exact, and by induction the accumulator converges to softmax(QK^T)V after all key blocks are processed. The single 1/l division is deferred to the very end.

For the backward pass, FlashAttention stores only the output O and per-row softmax statistics (a log-sum-exp value), plus the dropout RNG state. It recomputes S and P on-chip from blocks of Q, K, V and the saved statistics. This is selective gradient checkpointing, but because the saved cost is HBM traffic rather than FLOPs, the recomputation is faster than reading an N x N matrix from HBM. A key simplification keeps the backward pass block-wise: the length-N reduction D_i = P_i: dot dP_i: collapses to the length-d dot product dO_i dot O_i, since O_i is already known from the forward pass. Gradients dQ, dK and dV are then accumulated block by block without ever materializing P in HBM.

The result is exact attention with the same O(N^2 d) FLOPs as standard attention but only O(N) extra memory and O(N^2 d^2 / M) HBM accesses, where M is the SRAM capacity. Because d^2 is much smaller than M for typical head dimensions, this is many-fold fewer HBM accesses than the standard Theta(N^2), which is the source of the speedup. The block-tiling also lets the entire forward pipeline, matmul, masking, softmax and the second matmul, fuse into a single kernel with no intermediate spills.

```python
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q, K, V, Out, Lse,
    sm_scale,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_oh, stride_om, stride_ok,
    seqlen,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh
    lse_offset = off_hz * seqlen

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    qk_scale = sm_scale * 1.44269504  # log2(e)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    hi = tl.minimum((start_m + 1) * BLOCK_M, seqlen) if IS_CAUSAL else seqlen
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        qk = tl.where((start_n + offs_n)[None, :] < seqlen, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    lse_ptrs = Lse + lse_offset + offs_m
    tl.store(lse_ptrs, m_i + tl.math.log2(l_i), mask=offs_m < seqlen)
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < seqlen)


def attention_forward(q, k, v, causal=True, sm_scale=None):
    batch, nheads, seqlen, headdim = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(headdim)
    o = torch.empty_like(q)
    lse = torch.empty((batch * nheads, seqlen), device=q.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(seqlen, BLOCK_M), batch * nheads)
    _attn_fwd[grid](
        q, k, v, o, lse, sm_scale,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        o.stride(1), o.stride(2), o.stride(3),
        seqlen,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=headdim, IS_CAUSAL=causal,
    )
    return o, lse
```
