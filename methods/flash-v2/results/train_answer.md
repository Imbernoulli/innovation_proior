The problem is exact self-attention: for query, key, and value tensors of shape (N, d), we need O = softmax(Q K^T) V. A standard implementation materializes the N×N score and probability matrices in HBM, so it is memory-bound and quadratic in memory. The immediate predecessor, an IO-aware tiled attention kernel, already solved the memory side by streaming blocks of Q, K, and V through on-chip SRAM, using the online softmax recurrence to combine partial results, and never writing the N×N matrices to HBM. That gave a 2–4× speedup and linear memory. Yet on an A100 or H100 the forward pass still reaches only about 25–40% of the device's matmul peak, while optimized GEMMs run at 80–90%. Attention has become compute-underutilized, not memory-bound. Approximate attention methods are not the answer here because they change the computed function, and the goal is to speed up exact attention. The real target is therefore to keep the exact output and the linear memory guarantee, but reschedule the work so the Tensor Cores stay busy.

The reason scheduling matters is the GPU cost model: a non-matmul FLOP on the regular ALUs is roughly 16× more expensive than a Tensor-Core matmul FLOP. The exponentials, row maxes, divisions, rescalings, and causal-mask comparisons in attention are a small fraction of the raw FLOP count, but they can dominate wall-clock time because each one is paid for in the scarce non-matmul budget. The existing tiled kernel still pays several of these costs inside the hot loop: it rescales and divides the running output by the normalizer on every key block, it applies the causal mask to every block, it computes natural exp, and it parallelizes only over batch × heads, which leaves streaming multiprocessors idle when long sequences force a small batch. These are all scheduling problems, not mathematical problems.

I propose FlashAttention-2. It is an exact fused attention kernel that keeps the same output and memory guarantees as the IO-aware predecessor, but restructures the forward pass around three output-preserving levers. The first lever strips non-matmul work out of the inner loop. Instead of keeping a correctly normalized partial output and dividing by the normalizer on every block, the kernel keeps an unnormalized output accumulator plus a running normalizer and divides only once, after the last key block. The accumulator still has to be rescaled by the online-max correction whenever the running maximum increases, but that is unavoidable; the per-block division is not. The softmax scale and the log2(e) factor needed for base-2 exponent are folded into the query tile when it is loaded, so the inner loop computes only exp2, which the special-function unit handles faster than natural exp, with no extra per-element multiply. Causal masking is split into two passes: blocks entirely below the diagonal are processed with no mask at all, blocks entirely above the diagonal are skipped entirely, and only the one diagonal-crossing block per query row block applies the elementwise tl.where. These moves remove most of the expensive ALU operations from the bulk of the work without changing the result.

The second lever fixes occupancy. Softmax couples keys only within a single query row; different query rows are independent. FlashAttention-2 therefore makes the outer loop iterate over query-row blocks and parallelizes that outer loop across the sequence dimension in addition to batch and heads. Each row block becomes an independent thread block, so even when long sequences force a small batch there are enough blocks to fill all streaming multiprocessors. As a side benefit, the running output for a row block stays in SRAM and registers for the whole inner loop and never round-trips to HBM mid-computation. The third lever re-partitions work inside a thread block. Rather than splitting K and V across warps and sharing Q, which forces an inter-warp shared-memory reduction of partial results, FlashAttention-2 splits Q across warps and keeps K and V shared. Each warp owns a contiguous slice of rows, computes its own QK^T and PV slices using the shared tiles, and never needs another warp's partial result because rows do not couple. The block's output is simply the concatenation of the warps' slices. Finally, a single base-2 log-sum-exp statistic, m + log2(l), is stored per row for the backward pass instead of keeping both m and l separately.

The resulting forward algorithm is simple. For each query row block, load the pre-scaled query tile once. Loop over key/value column blocks, updating the running row maximum m, the running normalizer l, and the unnormalized accumulator acc. After the final block, divide acc by l once, compute the base-2 log-sum-exp, and store the output and the statistic. Block sizes of {64, 128} × {64, 128} are chosen per head dimension to trade fewer loop iterations against register and shared-memory pressure. Every change preserves exactness; only the schedule changes.

```python
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q, K, V, Out, M,
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

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    q = (q * (sm_scale * 1.44269504)).to(tl.float16)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    if IS_CAUSAL:
        non_causal_end = (start_m * BLOCK_M // BLOCK_N) * BLOCK_N
    else:
        non_causal_end = seqlen

    for start_n in range(0, non_causal_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        qk = tl.dot(q, tl.trans(k))
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    if IS_CAUSAL:
        hi = (start_m + 1) * BLOCK_M
    else:
        hi = non_causal_end

    for start_n in range(non_causal_end, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        qk = tl.dot(q, tl.trans(k))
        valid_cols = (start_n + offs_n[None, :]) < seqlen
        causal_cols = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = tl.where(valid_cols & causal_cols, qk, float("-inf"))
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    m_i = m_i + tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * seqlen + offs_m
    tl.store(m_ptrs, m_i, mask=offs_m < seqlen)
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen)


def custom_attention_forward(q, k, v, causal=True, sm_scale=None):
    batch, nheads, seqlen, headdim = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(headdim)
    o = torch.empty_like(q)
    m = torch.empty((batch, nheads, seqlen), device=q.device, dtype=torch.float32)
    if headdim <= 64:
        BLOCK_M, BLOCK_N = 128, 64
    elif headdim <= 128:
        BLOCK_M, BLOCK_N = 128, 64
    else:
        BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(seqlen, BLOCK_M), batch * nheads)
    _attn_fwd[grid](
        q, k, v, o, m, sm_scale,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        o.stride(1), o.stride(2), o.stride(3),
        seqlen,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=headdim, IS_CAUSAL=causal,
    )
    return o, m
```
