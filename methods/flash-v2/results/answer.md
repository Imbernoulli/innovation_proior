# FlashAttention-2, distilled

FlashAttention-2 is an exact (no-approximation) fused attention kernel that takes the
memory-IO-optimal tiled attention algorithm and re-schedules its work to keep the GPU's
Tensor Cores busy, closing the gap to GEMM-level utilization. It changes nothing about the
mathematical output; it changes where and how the computation is laid out across the device,
across thread blocks, and across the warps inside a block.

## Problem it solves

The previous fused attention kernel already removed the `O(N^2)` memory and most of the HBM
traffic of standard attention, but on an A100 its forward pass reaches only ~25-40% of the
device's matmul peak, while optimized GEMM reaches 80-90%. Attention has become
compute-underutilized rather than memory-bound. Goal: same exact output `O = softmax(QK^T)V`
and the same `O(N)` extra memory, but push utilization toward GEMM by fixing the work
scheduling — without any approximation.

## Key ideas (three levers, all output-preserving)

The cost model that drives everything: on Tensor-Core GPUs, matmul is ~16x cheaper per FLOP
than non-matmul (A100: 312 TFLOPs/s FP16/BF16 matmul vs 19.5 TFLOPs/s non-matmul FP32). So
non-matmul operations, though a small fraction of FLOPs, dominate the time and are the thing
to remove.

1. **Reduce non-matmul FLOPs in the inner loop.**
   - *Defer normalization.* Don't divide the running output by the normalizer every block;
     keep an **unnormalized** output accumulator `Õ` plus the running normalizer `ℓ`, and
     divide once at the very end: `O = diag(ℓ^last)^{-1} Õ^last`. Per block the accumulator is
     only rescaled by the online-max correction `diag(e^{m^{(j-1)} - m^{(j)}})` (unavoidable),
     not divided. This collapses `T_c` per-block divisions into one.
   - *Store only the logsumexp* for the backward pass (one vector, not two): `L = m + log(ℓ)`
     in the natural-exponential math, or `M = m + log2(ℓ)` in the base-2 kernel. Backward
     recomputes `P = exp2(S - M)` when the scores are in base-2 units.
   - *Use `exp2` not `exp`.* GPU special-function units compute base-2 exponent natively/faster;
     `e^x = 2^{x·log2(e)}`, `log2(e) = 1.44269504`. Fold `log2(e)` together with the softmax
     scale and multiply them onto `Q` at load time, so the hot loop is pure `exp2` with no
     per-element scale multiply.
   - *Two-pass causal masking.* For each query row block: blocks entirely above the diagonal
     are skipped outright (~half the blocks for long sequences, ~1.7-1.8x); blocks strictly
     below the diagonal need no element mask; only the one on-diagonal block per row block uses
     `tl.where`. Split the key loop into a no-mask pass and a small boundary pass.

2. **Parallelize over sequence length (loop swap).** Query rows are independent (softmax
   couples keys within a row, not rows to each other). Make the **outer loop run over query row
   blocks** and the inner loop over K/V column blocks, and launch the grid over
   (row blocks) × (batch) × (heads). Each row block becomes an independent thread block, so the
   108 SMs stay full even when long sequences force a small batch; the running output also never
   round-trips to HBM mid-loop.

3. **Split-Q warp partitioning.** Inside a thread block, split **Q** across the (4 or 8) warps
   and keep **K, V** shared by all of them — instead of the old "split-K" (split K,V, share Q).
   Each warp owns a slice of rows, computes its `QK^T` against shared `K` and its output against
   shared `V`, and never needs another warp's partial, since rows don't couple. This removes the
   inter-warp shared-memory reduction (write partials → sync → sum) that split-K forced.

**Block sizes** `{64,128} × {64,128}` chosen per head dimension `d`: larger blocks cut
shared-memory traffic but raise register/shared-memory pressure (and large `d` like 256 must
shrink the row block to avoid register spilling / shared-memory overflow). Tuned per head dim;
autotuning is a natural extension.

## Forward algorithm

For each query row block `i` (its own thread block), with unnormalized accumulator `Õ_i`,
running max `m_i = -∞`, running normalizer `ℓ_i = 0`:

```
load Q_i once (pre-scaled by sm_scale·log2(e))
for each K/V column block j in the loop (two passes if causal):
    S = Q_i K_j^T                              # already in log2 scale
    if on-diagonal causal block: S = where(row >= col, S, -inf)
    m_new   = max(m_i, rowmax(S))
    alpha   = exp2(m_i - m_new)                # max-correction factor
    P       = exp2(S - m_new)                  # exp2 on the SFU
    ℓ_i     = ℓ_i · alpha + rowsum(P)
    Õ_i     = Õ_i · alpha + P V_j              # accumulate UNNORMALIZED
    m_i     = m_new
O_i = Õ_i / ℓ_i                                # normalize ONCE at the end
M_i = m_i + log2(ℓ_i)                          # base-2 logsumexp, for backward
```

Returns exactly `O = softmax(QK^T)V`, in `O(N^2 d)` FLOPs and `O(N)` extra memory (the stored
logsumexp) — the same correctness and memory guarantees as the IO-aware predecessor, scheduled to
keep the Tensor Cores busy. Causal `O(N^2 d / 2)` FLOPs since ~half the score matrix is skipped.

## Working code (Triton forward pass)

A Triton-style forward pass returns the output and the base-2 logsumexp needed by the backward.

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
    # One program = one query row block of one head.
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Q loaded once; scale + log2(e) folded into the tile before QK^T.
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    q = (q * (sm_scale * 1.44269504)).to(tl.float16)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")   # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0            # first alpha=0 makes this a zero seed
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)    # UNNORMALIZED output

    # Causal -> two passes; upper-triangle blocks never iterated.
    if IS_CAUSAL:
        # floor to a BLOCK_N multiple so pass 1 covers only whole below-diagonal blocks
        non_causal_end = (start_m * BLOCK_M // BLOCK_N) * BLOCK_N
    else:
        non_causal_end = seqlen

    # Pass 1: no-mask blocks (strictly below the diagonal; all columns valid).
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

    # Pass 2: boundary block(s) on the diagonal (mask applied).
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

    # Normalize once, then store output plus base-2 logsumexp.
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
    # Per-headdim block sizes.
    if headdim <= 64:
        BLOCK_M, BLOCK_N = 128, 64
    elif headdim <= 128:
        BLOCK_M, BLOCK_N = 128, 64
    else:
        BLOCK_M, BLOCK_N = 64, 64
    # Grid: parallel over row blocks (sequence length) AND batch*heads.
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

## Relation to prior methods

- **IO-aware tiled attention (predecessor):** same exact output and linear memory, but its
  forward loops outer-over-K/inner-over-Q (so the output round-trips to HBM), parallelizes only
  over batch×heads (low occupancy on long sequences), normalizes the output every block (extra
  non-matmul divides), and uses split-K warp partitioning (shared-memory reduction). This method
  swaps the loops, parallelizes over sequence length, defers normalization, and uses split-Q.
- **Online softmax (Milakov & Gimelshein 2018):** the single-pass running-max/normalizer
  recurrence `d_j = d_{j-1} e^{m_{j-1}-m_j} + e^{x_j-m_j}` and its associative block-merge
  operator are the engine that lets attention be computed block-by-block.
- **Memory-efficient attention (Rabe & Staats 2021):** the "accumulate the output unnormalized,
  divide once at the end" idea reused to remove the per-block division.
- **Approximate attention (Reformer/Performer/Longformer/etc.):** drop below quadratic cost by
  approximating; this method keeps exact attention and instead optimizes its scheduling.
