Standard exact-attention kernels are already arithmetically minimal, yet on an NVIDIA Hopper H100 the best fused implementation reaches only about thirty-five percent of the chip's theoretical peak FLOPs, while a well-tuned matrix multiply hits eighty to ninety percent. The bottleneck is not the number of operations. Attention is two large GEMMs separated by a row-wise softmax, and those two phases run on very different hardware: the Tensor Cores deliver roughly a petaflop of FP16 matmul throughput, while the special-function unit that computes exponentials is about two hundred and fifty-six times slower. In a synchronous schedule the kernel issues the first matmul, waits, exponentiates, waits, then issues the second matmul, so the fast unit sits idle while the slow unit runs and vice versa. For a typical head dimension of one hundred and twenty-eight the exponential can consume on the order of half the matmul's wall time, which is exactly why the overall utilization collapses. Moving to FP8 makes the imbalance worse, because matmul throughput doubles while exponential throughput does not.

FlashAttention and FlashAttention-2 solved the memory problem by tiling Q, K, and V through on-chip SRAM and applying the online-softmax recurrence block by block, never writing the N-by-N score or probability matrices to HBM. But their schedule is still effectively synchronous and sixteen-bit only, so they leave Hopper's asynchronous Tensor Core and Tensor Memory Accelerator units unused. The next step, therefore, is not a new attention formula but a new schedule: keep the exact softmax(QK^T)V computation, but overlap the independent loads, matmuls, and exponentials across the hardware units that can run concurrently, and shave every avoidable bit of non-matmul work.

The method is FlashAttention-3. It exploits deep hardware asynchrony at three levels. First, producer-consumer warp specialization splits a threadblock into load warps and compute warps: the producers issue asynchronous TMA copies of the next K and V tiles into a circular shared-memory buffer while the consumers issue WGMMA matmuls on the current tiles. Registers are reallocated toward the consumers with setmaxnreg, so memory latency hides under compute. Second, inter-warpgroup pingpong scheduling uses barriers to keep one warpgroup's matmuls on the Tensor Cores while another warpgroup's softmax runs on the special-function unit, then swaps their roles, so neither unit idles. Third, a two-stage intra-warpgroup pipeline breaks the per-iteration chain S=QK^T, softmax, PV by keeping both a current and a next score tile in registers; the second matmul of one iteration overlaps the softmax of the next. Two stages are the sweet spot: a third stage adds register pressure without reliably overlapping the second matmul.

Orthogonally, FlashAttention-3 reduces the cost of the softmax itself. The scale factor one over root d is multiplied once by log2(e), so every exponential becomes a hardware exp2 instruction via the identity e^x = 2^(x log2 e). The output accumulator is left un-normalized throughout the loop and divided by the final normalizer once at the end; only the logsumexp L = m + log(l) is stored for reconstruction. Causal masking is split into two passes: blocks entirely below the diagonal are processed with no mask at all, future blocks are skipped entirely, and the elementwise where mask is applied only to the diagonal-crossing boundary blocks. For raw matmul throughput the two GEMMs can run in FP8, with in-kernel transposition of V tiles and a byte-permute of the FP32 accumulator to satisfy FP8 layout constraints. Accuracy under FP8 is protected by block-wise quantization, since the algorithm already works in tiles, and by incoherent processing: Q and K are rotated by a normalized Hadamard matrix with random sign flips before quantization. Because the rotation matrix M satisfies M M^T = I, the attention scores are unchanged exactly, but outliers are spread across many entries so per-block scales no longer crush ordinary values.

The online-softmax recurrence guarantees that all of this still computes softmax(QK^T)V exactly, regardless of how blocks are partitioned or when the rescaling is applied. The exp2 substitution and the incoherent rotation are exact identities; the only approximation is the FP8 operand quantization itself, and its error is controlled by the per-block and Hadamard scaling. The result is a kernel that keeps the chip's compute units fed rather than taking turns.

```python
import math
import torch
import triton
import triton.language as tl

LOG2E = 1.44269504  # log2(e): e^x = 2^(x*log2 e), so exp2 replaces exp.


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32},  num_stages=4, num_warps=4),
    ],
    key=['seqlen', 'BLOCK_DMODEL', 'IS_CAUSAL', 'warp_specialize'],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, Out, L,
    sm_scale,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_oh, stride_om, stride_ok,
    seqlen,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, IS_CAUSAL: tl.constexpr,
    warp_specialize: tl.constexpr,
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

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * LOG2E

    if IS_CAUSAL:
        non_causal_end = (start_m * BLOCK_M // BLOCK_N) * BLOCK_N
    else:
        non_causal_end = seqlen

    for start_n in tl.range(0, non_causal_end, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        qk = tl.dot(q, tl.trans(k))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1) * qk_scale)
        qk = qk * qk_scale - m_new[:, None]
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    if IS_CAUSAL:
        hi = (start_m + 1) * BLOCK_M
        for start_n in tl.range(non_causal_end, hi, BLOCK_N, warp_specialize=warp_specialize):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
            k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
            qk = tl.dot(q, tl.trans(k))
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0.0, -1.0e6)
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            qk -= m_new[:, None]
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
            acc = tl.dot(p.to(v.dtype), v, acc)
            m_i = m_new

    l_ptrs = L + off_hz * seqlen + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=offs_m < seqlen)

    acc = acc / l_i[:, None]
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen)


def custom_attention_forward(q, k, v, causal=True, sm_scale=None):
    batch, nheads, seqlen, headdim = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(headdim)
    o = torch.empty_like(q)
    lse = torch.empty((batch, nheads, seqlen), device=q.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(seqlen, META['BLOCK_M']), batch * nheads)
    _attn_fwd_kernel[grid](
        q, k, v, o, lse, sm_scale,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        o.stride(1), o.stride(2), o.stride(3),
        seqlen,
        BLOCK_DMODEL=headdim, IS_CAUSAL=causal, warp_specialize=True,
    )
    return o
```
