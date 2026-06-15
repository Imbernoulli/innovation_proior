# Context: making exact attention compute-bound on Tensor-Core GPUs (circa 2022-2023)

## Research question

Self-attention is the runtime and memory bottleneck of Transformers, and both grow
quadratically in the sequence length `N`. The demand for long context is rising fast — models
are pushing from the old 2k limit toward 32k, 64k, even 100k tokens — so the cost of attention
is exactly what gates training and serving longer sequences. An IO-aware tiled kernel (the
immediate prior art described below) has already solved the *memory* side: it computes exact
attention with memory linear in `N` instead of quadratic, and cuts high-bandwidth-memory (HBM)
traffic enough to run 2-4x faster than a standard implementation, with no approximation. The
remaining problem is throughput. On an A100, an optimized matrix-multiply (GEMM) reaches
80-90% of the device's theoretical peak FLOPs/s, but the tiled attention kernel reaches only
about 25-40% of peak on the forward pass (and worse on the backward). Attention is no longer
memory-bound — it is *compute-underutilized*. The precise goal: keep the exact output and the
linear memory of the tiled kernel, but close the gap to GEMM-level utilization on a modern
Tensor-Core GPU, where the underused resource is the matrix-multiply units and the scarce
budget is everything that is *not* a matrix multiply. A solution has to find where the kernel's
work is being scheduled badly — across the device, across thread blocks, and across the warps
inside a block — and fix the scheduling without touching the mathematical result.

## Background

**The GPU execution and performance model.** A modern GPU is a sea of streaming
multiprocessors (SMs) — 108 of them on an A100. A kernel is launched as a grid of thread
blocks; each thread block is scheduled onto one SM and runs to completion there. Within a
thread block, threads are grouped into warps of 32, and warps within a block communicate only
by writing to and reading from on-chip shared memory (with a synchronization barrier in
between); threads within a warp can communicate by fast register shuffles. The memory
hierarchy is steeply asymmetric: HBM is large but slow (40-80 GB at 1.5-2.0 TB/s on A100),
on-chip SRAM / shared memory is tiny but fast (192 KB per SM, roughly 19 TB/s). Two
consequences dominate everything that follows. First, *occupancy*: if a kernel launches fewer
thread blocks than there are SMs (or not enough to hide latency), some SMs sit idle and the
device is underused. Second, and decisive here, *matmul is far cheaper than non-matmul*. Nvidia
GPUs have dedicated Tensor Cores for low-precision matrix multiply: an A100 does 312 TFLOPs/s
of FP16/BF16 matmul but only 19.5 TFLOPs/s of non-matmul FP32. Each non-matmul FLOP — every
exponential, every max, every divide, every rescale — is therefore about **16x more
expensive** than a matmul FLOP. Even though the non-matmul operations in attention are a small
*fraction* of the FLOP count, they can dominate the *time*, so any operation that is not a
matrix multiply is suspect and worth removing.

**Standard attention and why it is slow.** Given `Q, K, V ∈ R^{N×d}` (sequence length `N`,
head dimension `d`, typically `N` ≈ 1k-8k and `d` ≈ 64-128), attention computes
`S = QK^T ∈ R^{N×N}`, `P = softmax(S)` row-wise, `O = PV ∈ R^{N×d}`. The standard
implementation calls a GEMM for `S`, writes the `N×N` matrix to HBM, reads it back to compute
the softmax, writes `P` to HBM, then calls a GEMM for `O`. It materializes the two `N×N`
matrices, so it needs `O(N^2)` memory and `O(Nd + N^2)` HBM accesses; since softmax and the
read/write traffic are memory-bound, wall-clock time is dominated by HBM, not arithmetic. The
backward pass additionally has to save `P` (`N×N`) to compute gradients.

**Numerically stable softmax costs passes.** The softmax of a vector `x` must subtract the row
max before exponentiating, or `e^{x_i}` overflows (already at `x_i ≳ 89` in float32/bfloat16).
The "safe" softmax therefore reads the input three times — one pass to find the max `m`, one to
accumulate the normalizer `d = Σ e^{x_j - m}`, one to emit `y_i = e^{x_i - m}/d` — four memory
accesses per element. This three-pass structure is the cost that motivated computing the max
and the normalizer *together*.

**Online normalizer calculation (Milakov & Gimelshein, 2018).** The max and the normalizer can
be computed in a single pass with a running correction. Iterating over the elements, keep a
running max `m_j` and a running normalizer `d_j`:
`m_j = max(m_{j-1}, x_j)`, `d_j = d_{j-1} · e^{m_{j-1} - m_j} + e^{x_j - m_j}`. The factor
`e^{m_{j-1} - m_j}` retroactively rescales the accumulated normalizer whenever the running max
increases, so a single sequential pass yields exactly the safe-softmax `m` and `d` (proved by
induction). They further note that the update on the pair `(m, d)` is an *associative and
commutative* binary operation, `(m_a,d_a) ⊕ (m_b,d_b) = (max(m_a,m_b), d_a e^{m_a-max} +
d_b e^{m_b-max})`, so the running statistics of two arbitrary *chunks* can be merged — the
softmax of a long vector can be assembled from per-chunk summaries in any order. (The idea
descends from Welford's 1962 online corrected-sum recurrence.)

**Deferring the normalization (Rabe & Staats, 2021).** For attention specifically, the
division by the normalizer can be moved to the very end. By the distributive law,
`attention(q, K, V) = (Σ_i v_i e^{s_i}) / (Σ_j e^{s_j})` with `s_i = q·k_i`, so one can keep an
*unnormalized* output accumulator `v*` and a scalar normalizer `s*`, both initialized to zero,
update them as keys/values stream in, and divide `v*/s*` only once at the end. Combined with the
online running max `m*`, the stable updates are
`v* ← v* e^{m*_old - m_i} + v_i e^{s_i - m_i}` and `s* ← s* e^{m*_old - m_i} + e^{s_i - m_i}`.
Processing keys/values in `√N`-sized chunks gives `O(√N)` memory; the chunk size trades runtime
against memory and is left to the programmer. The takeaway carried forward: accumulate the
output unnormalized and normalize once at the end.

## Baselines

**Standard (materialized) attention.** Core idea and math as above: `S = QK^T`, `P =
softmax(S)`, `O = PV`, computed as three separate GPU operations with the `N×N` matrices
written to and read from HBM. **Gap:** `O(N^2)` memory and `O(Nd + N^2)` HBM accesses make it
memory-bound and quadratic-memory; it cannot reach long sequence lengths and leaves the Tensor
Cores idle while waiting on HBM.

**IO-aware tiled attention (Dao, Fu, Ermon, Rudra & Ré, 2022).** This is the direct
predecessor — exact attention fused into a single GPU kernel using tiling and recomputation.
It splits `Q, K, V` into blocks, loads blocks from HBM into SRAM, and computes attention
block-by-block, using the online-softmax statistics `(m, ℓ)` to combine blocks and rescale the
running output so the final result is exact (no approximation). It never writes the `N×N`
matrices `S` or `P` to HBM. For the backward pass it stores only `O` and the softmax
statistics and *recomputes* `S, P` from blocks in SRAM (selective gradient checkpointing), so
its memory is linear in `N`. Its IO complexity is `Θ(N^2 d^2 / M)` HBM accesses (with `M` the
SRAM size), versus `Θ(Nd + N^2)` for standard attention; since `d^2 ≪ M` for typical `d` and
`M`, this is many-fold fewer accesses, and it is provably optimal up to constants over a range
of `M`. It runs 2-4x faster than standard attention with 10-20x memory savings. Its concrete
forward structure: the **outer loop runs over blocks of `K` and `V` (columns)** and the inner
loop over blocks of `Q` (rows); each inner step loads the running output block `O_i` and its
statistics `(m_i, ℓ_i)` from HBM, applies the online-softmax update
`O_i ← diag(ℓ_i^{new})^{-1}(diag(ℓ_i) e^{m_i - m_i^{new}} O_i + e^{m̃_{ij} - m_i^{new}} P̃_{ij}
V_j)`, and writes `O_i, m_i, ℓ_i` back to HBM. It is parallelized over the batch and head
dimensions — one thread block per attention head — and inside each thread block it uses a
"split-K" warp scheme: `K` and `V` are split across the (typically 4) warps while `Q` is shared
by all of them. **Gaps (all observed limitations, all in how the work is laid out, not in the
result):**
- The forward online-softmax update *rescales the running output by the normalizer on every
  block* (the `diag(ℓ_i^{new})^{-1}` factor) and keeps both `m_i` and `ℓ_i` around, so each
  inner step pays several non-matmul operations — divides and rescales — and these are the
  16x-expensive kind.
- With the outer loop over `K/V`, the running output block `O_i` and its statistics are read
  from and written to HBM once per column block, and the parallelism is only over batch × heads
  (one thread block per head). When the sequence is long the batch is usually small, so
  batch × heads can fall below the number of SMs, leaving SMs idle — low occupancy.
- The split-K warp scheme makes each warp compute a partial `QK^T` and a partial product with
  `V`; the partials from different warps must be written to shared memory, synchronized, and
  summed before the block's output is complete. These shared-memory round-trips inside every
  thread block are pure overhead.

**Approximate-attention methods (Reformer, Performer, Linformer, Longformer, BigBird, and
linear-attention variants, 2020-2021).** Core idea: replace the dense `N×N` softmax attention
with a sparse, low-rank, or kernelized approximation to drop the asymptotic cost below
quadratic. **Gap:** they change the computed function (they approximate), and despite their
existence most large-scale training runs still use exact attention — so they do not address the
goal of speeding up *exact* attention.

## Evaluation settings

The natural yardsticks, all pre-existing:
- **Attention micro-benchmark on A100.** Forward (and backward) of multi-head attention at the
  shapes used in practice: head dimensions `d ∈ {64, 128}` (and larger), sequence lengths from
  ~1k to ~16k, with and without a causal mask, in FP16/BF16. The headline metric is achieved
  TFLOPs/s and its ratio to the device's theoretical matmul peak (the "% of peak" that GEMM
  reaches 80-90% of). Latency (ms) per forward pass is the companion metric.
- **FLOP accounting.** Attention forward FLOPs counted as `4 · batch · N^2 · heads · d`
  (the two GEMMs, `QK^T` and `PV`), **halved when causal** since roughly half the score matrix
  is masked. Throughput is FLOPs divided by measured time.
- **Correctness.** Maximum absolute difference of the output against a reference exact-attention
  implementation must stay below a small tolerance — the kernel must return the exact result,
  not an approximation.
- **End-to-end training.** GPT-style language models trained on standard text corpora,
  reporting model-FLOPs-utilization (achieved training TFLOPs/s per GPU as a fraction of peak).
- Protocol: fixed total token count across shapes so configurations are comparable; identical
  inputs and scale across the kernel under test and the reference.

## Code framework

The substrate is a GPU tiling DSL (Triton-style) in which one writes a `@jit` kernel that runs
as a grid of programs, each program loading tiles of the inputs from HBM into on-chip SRAM,
computing on them, and storing tiles of the output back — plus a thin Python wrapper that
allocates the output, picks the launch grid and the tile (block) sizes, and launches the
kernel. What already exists: the tiling/loading primitives (`tl.load`/`tl.store` with strided
pointers and boundary masks, `tl.dot` for the Tensor-Core matmul, the `tl.max`/`tl.sum`
reductions, and the on-chip `exp`), the online-softmax recurrence on running statistics, and
the fact that one program can own a tile of the problem and loop over the contraction
dimension. What is *not* settled — and is exactly what is to be designed — is **how the
attention computation is mapped onto the grid, the per-block loop, and the warps within a
block**: which dimension the outer loop and the launch grid range over, how the online-softmax
update is arranged to minimize non-matmul work, and how a block's work is divided among its
warps. The scaffold is therefore a bare fused-attention harness with one empty kernel body and
an unfilled grid/block choice.

```python
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q, K, V, Out, Stats,
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
    """Fused exact-attention forward. One program owns part of the problem and
    loops over the contraction (key/value) dimension, keeping the softmax
    running statistics in registers/SRAM, writing the per-row statistic needed
    by backward, and never materializing the N x N score matrix in HBM.

    Available primitives (already exist): tl.load/tl.store with strided
    pointers + boundary masks, tl.dot (Tensor-Core matmul), tl.max / tl.sum
    reductions, on-chip exp, and the online-softmax recurrence on (m, l).
    """
    # TODO: the work mapping and inner-loop schedule we will design --
    #       which dimension the program grid / outer loop ranges over, how the
    #       online-softmax update is arranged, and how the block's work is
    #       partitioned. Compute the exact attention output O = softmax(QK^T) V
    #       (with optional causal mask), and store O plus the row statistic.
    pass


def custom_attention_forward(q, k, v, causal=True, sm_scale=None):
    """Wrapper: allocate output, choose the launch grid and block sizes, launch
    the fused kernel. q, k, v: (batch, nheads, seqlen, headdim), contiguous, FP16."""
    batch, nheads, seqlen, headdim = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(headdim)
    o = torch.empty_like(q)
    stats = torch.empty((batch, nheads, seqlen), device=q.device, dtype=torch.float32)
    # TODO: the grid and block sizes we will choose -- the launch grid decides
    #       how the problem is spread across the GPU's streaming multiprocessors;
    #       BLOCK_M x BLOCK_N is the tile each program works on.
    BLOCK_M, BLOCK_N = ..., ...
    grid = ...
    _attn_fwd[grid](
        q, k, v, o, stats, sm_scale,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        o.stride(1), o.stride(2), o.stride(3),
        seqlen,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=headdim, IS_CAUSAL=causal,
    )
    return o, stats
```

The kernel body and the grid/block choice are the empty slots; everything else — the loading
primitives, the matmul, the reductions, the running-statistics recurrence — already exists.
