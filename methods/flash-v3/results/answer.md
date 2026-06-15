# FlashAttention-3, distilled

FlashAttention-3 is a fused exact-attention kernel that redesigns the tiled, online-softmax
attention algorithm to exploit two capabilities of a modern asynchronous GPU (Hopper): deep
hardware **asynchrony** (the Tensor Core and the memory-copy unit run concurrently and
non-blocking) and cheap **low precision** (FP8 matmul at ~2x FP16 throughput). It keeps the
exact `softmax(QK^T)V` computation of its predecessors but overlaps the work so the fast and
slow compute units stop taking turns.

## Problem it solves

Self-attention is matmul-dominated yet the best prior fused kernel reaches only ~35% of an
H100's peak FLOPs (vs 80-90% for an optimized GEMM). The bottleneck is not arithmetic: the
algorithm serializes matmul (Tensor Cores, ~989 TFLOPS FP16) against the softmax exponential
(special-function unit, ~3.9 TFLOPS — a ~256x throughput gap), so each unit idles while the
other works. For a head-dim-128 forward pass the exponential can take ~50% of the matmul's
wall time; FP8 makes the imbalance worse, since matmul doubles and the exponential does not.

## Key idea

Overlap the asynchronous units at three grains, and cut the slow-side cost directly:

1. **Producer-consumer asynchrony (warp specialization).** Split a threadblock's warps into
   producers that issue only async memory loads (TMA) and consumers that issue only matmuls
   (WGMMA), with an `s`-stage circular SMEM buffer so producers run ahead; reallocate registers
   to the consumers (`setmaxnreg`). Hides memory latency under compute.
2. **Pingpong scheduling (inter-warpgroup).** Use barriers to force one warpgroup's matmuls
   before another's, so one warpgroup runs softmax (on the special-function unit) while the
   other runs matmul (on the Tensor Cores), then swap. Keeps both units busy.
3. **2-stage GEMM-softmax pipeline (intra-warpgroup).** Break the per-iteration dependency
   `S=QK^T → softmax → PV` by pipelining across iterations with a register buffer
   (`S_cur`, `S_next`): iteration `j`'s second matmul `P_cur V_{j-1}` overlaps iteration
   `j+1`'s softmax. Costs one extra `S` tile (`B_r·B_c·4` bytes) in registers, so depth is a
   profiled trade-off; a 3-stage variant is worse (more registers; compiler won't overlap the
   second matmul).
4. **Cheaper softmax:** hardware `exp2` (`e^x = 2^{x·log2 e}`, fold `log2 e=1.44269504` into
   the score scale once); an un-normalized output accumulator divided by the normalizer only
   once at the end; store only the logsumexp `L=m+log ℓ`; two-pass causal (skip future blocks,
   mask only the diagonal-crossing boundary blocks).
5. **FP8 matmul (2x).** Solve the layout constraints — FP8 WGMMA needs k-major operands, so
   transpose `V` tiles in-kernel via LDSM/STSM (hidden under prior matmuls), and byte-permute
   the FP32 accumulator to match the second GEMM's operand layout. Protect accuracy with
   **block quantization** (one scale per `B_r×d`/`B_c×d` tile — free, since the algorithm is
   block-wise) and **incoherent processing** (rotate `Q,K` by `M`=normalized
   Hadamard·random-±1; since `M M^T=I`, `(QM)(KM)^T=QK^T` is exact, but outliers
   are spread out; `O(d log d)`, fused into rotary).

## Why it is still exact

Online-softmax recurrence, per row, with running max `m`, normalizer `ℓ`, un-normalized
output `Õ`:
```
m'  = max(m, rowmax(s))
ℓ'  = e^{m-m'} · ℓ + rowsum(e^{s-m'})
Õ'  = e^{m-m'} · Õ + e^{s-m'} · v
O   = Õ_last / ℓ_last
```
Telescoping two blocks gives `Õ_last = Σ_j e^{s_j - m} v_j` and `ℓ_last = Σ_j e^{s_j - m}` with
`m` the true overall max, so `O = softmax(s)·V` exactly, for any block partition and any number
of max jumps. `exp2` is an exact identity; the incoherent rotation is exact; deferring the
`1/ℓ` only moves the single division. Only FP8 operand quantization is approximate, and its
error is mitigated by block and incoherent scaling.

## Forward pass algorithm (CTA view, 2-stage pipelined consumer)

```
Require: Q_i ∈ R^{B_r×d}, K,V ∈ R^{N×d} in HBM; key block size B_c, T_c = ceil(N/B_c).
Init s-stage circular SMEM buffer.
Producer warpgroup: deallocate regs; load Q_i; for j in 0..T_c-1: wait stage (j%s) free,
  TMA-load K_j,V_j into stage (j%s), signal consumer.
Consumer warpgroup: reallocate regs; O_i=0, ℓ_i=0, m_i=-inf.
  wait Q_i,K_0; S=Q_i K_0^T (commit,wait); compute m_i,P̃,ℓ_i from S, rescale O_i.
  for 1 ≤ j < T_c-1:
    wait K_j;     S_next = Q_i K_j^T            (commit, do NOT wait)   # GEMM-0
    wait V_{j-1}; O_i = O_i + P̃_cur V_{j-1}      (commit, do NOT wait)   # GEMM-1
    wait GEMM-0;  compute m_i,P̃_next,ℓ_i from S_next  # softmax overlaps the two GEMMs
    wait GEMM-1;  rescale O_i; release buffer stages; S_cur ← S_next
  wait V_{T_c-1}; O_i = O_i + P̃_last V_{T_c-1} (commit,wait)
  Epilogue: O_i = O_i / ℓ_i; L_i = m_i + log(ℓ_i); write O_i, L_i to HBM.
```

## Working code (tile-based DSL realization)

In a tile-based DSL, the forward path realizes the exact-math pieces directly: the score scale
is converted once to base 2, the inner loop uses `exp2`, the accumulator stays un-normalized
until the epilogue, and the causal path is split into an off-band pass plus a masked boundary
pass. The launch is autotuned over tile sizes, pipeline depth, and warp count; the low-level
Hopper warpgroup schedule and FP8 layout transforms are the hand-written-kernel layer beneath
this DSL form.

```python
import math
import torch
import triton
import triton.language as tl

LOG2E = 1.44269504  # log2(e): e^x = 2^(x*log2 e), so the loop uses hardware exp2.


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

    # qk_scale (below) folds the softmax scale with log2(e) so the loop's exp2
    # computes e^(sm_scale * score) using the hardware base-2 intrinsic.
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)  # un-normalized output
    qk_scale = sm_scale * LOG2E

    # Two-pass causal: no-mask blocks below the diagonal; future blocks skipped.
    if IS_CAUSAL:
        non_causal_end = (start_m * BLOCK_M // BLOCK_N) * BLOCK_N
    else:
        non_causal_end = seqlen

    for start_n in tl.range(0, non_causal_end, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        qk = tl.dot(q, tl.trans(k))                # GEMM-0
        m_new = tl.maximum(m_i, tl.max(qk, axis=1) * qk_scale)
        qk = qk * qk_scale - m_new[:, None]
        alpha = tl.math.exp2(m_i - m_new)          # rescale factor (hardware exp2)
        p = tl.math.exp2(qk)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)        # GEMM-1, no per-block 1/l
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

    # Store the base-2 logsumexp for exact on-chip probability reconstruction.
    l_ptrs = L + off_hz * seqlen + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=offs_m < seqlen)

    acc = acc / l_i[:, None]                        # single deferred division
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
