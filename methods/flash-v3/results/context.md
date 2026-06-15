# Context: fused exact-attention kernels on a new asynchronous GPU (circa 2024)

## Research question

Self-attention is the computational bottleneck of Transformer models: for a sequence of
length `N` and head dimension `d`, forming the score matrix `S = QK^T`, applying a row-wise
softmax, and multiplying by `V` costs `O(N^2 d)` FLOPs and, in a naive implementation,
`O(N^2)` extra memory. The memory pressure has already been solved by tiled, IO-aware
kernels that never materialize the `N x N` matrices, and a better-parallelized successor
raised throughput further. The open problem is a throughput one on the newest datacenter
GPU (the NVIDIA Hopper H100): the best available exact-attention kernel reaches only about
35% of the device's theoretical peak FLOPs, whereas a well-tuned matrix-multiply (GEMM)
kernel on the same chip reaches 80-90%. The goal is to close that gap — to design a fused
forward attention kernel that keeps the chip's compute units near saturation
— while remaining numerically faithful (max-abs error against a high-precision reference must
stay tiny) and supporting the standard causal mask. The pain point is not arithmetic count;
the kernel already does the minimal, matmul-dominated FLOPs. The pain point is that the
chip's specialized units sit idle for much of the run.

Two facts about the new hardware reframe what "fast" means here. First, the chip is deeply
*asynchronous*: matrix multiply runs on Tensor Cores driven by an asynchronous warpgroup-wide
instruction (WGMMA), memory movement between global and shared memory runs on a separate
dedicated asynchronous unit (the Tensor Memory Accelerator, TMA), and these issue without
blocking. Second, the chip offers cheap *low precision*: an 8-bit floating point format (FP8,
e4m3) runs matmul at roughly double the FP16/BF16 throughput. An attention algorithm written
for the previous, effectively synchronous, FP16 world uses neither — it follows a synchronous
schedule and runs only in 16-bit. A solution would have to put both of these new hardware
capabilities to work for attention while remaining numerically faithful.

## Background

**The arithmetic of attention.** For one head, with `Q, K, V ∈ R^{N×d}`, attention computes
`S = α QK^T`, `P = softmax(S)` row-wise, `O = PV`, with scale `α = 1/√d`. For numerical
stability one subtracts the row max from `S` before exponentiating. Multi-head attention runs
this independently per head and per batch element, so the whole thing is embarrassingly
parallel across `(batch × heads)` and, with the right loop structure, across query positions.
The backward pass follows from the chain rule: with `dO` the output gradient,
`dV = P^T dO`, `dP = dO V^T`, `dS = dsoftmax(dP)` where for a softmax row `p` the Jacobian is
`(diag(p) − p p^T)`, then `dQ = α dS K` and `dK = α dS^T Q`.

**The GPU memory and execution hierarchy.** Memory is a hierarchy with capacity inversely
related to bandwidth: off-chip global memory (HBM) — on the H100 SXM5, 80 GiB at ~3.35 TB/s —
an L2 cache, a small programmer-managed on-chip shared memory (SMEM, ~228 KiB per streaming
multiprocessor at tens of TB/s), and the per-thread register file. The execution model nests
threads → warps (32 threads) → warpgroups (4 warps) → threadblocks (CTAs) → clusters → grid;
threads in a CTA share SMEM and are co-scheduled on one SM. Operations are classified by
*arithmetic intensity* (FLOPs per byte) as compute-bound (large matmuls) or memory-bound
(elementwise ops, softmax, masking, normalization). Because compute has outpaced memory
bandwidth for years, fused kernels that load once from HBM and do all the work on-chip are
the standard remedy for memory-bound work.

**Two new hardware capabilities.** On Hopper, the Tensor Core is exposed through the
asynchronous WGMMA instruction, which can source its operands directly from SMEM and does not
block the issuing warp. The TMA unit performs asynchronous bulk copies between HBM and SMEM,
needs only a single thread to issue, and frees registers that index arithmetic would
otherwise consume. Hardware asynchrony makes *warp-specialized* kernels possible — partition
a CTA's warps into roles, some issuing only data movement and some only computation — which
generically improves the compiler's ability to find a good instruction schedule
(Bauer et al. 2011). Hopper can also dynamically reallocate registers between warpgroups
(`setmaxnreg`), giving the compute warps a larger share. Separately, FP8 WGMMA delivers about
2x the per-SM throughput of FP16, but with only 3 mantissa and 4 exponent bits, and with
strict operand-layout constraints (FP8 WGMMA accepts only operands contiguous in the inner
contraction dimension, and the FP32 accumulator layout does not match the operand-A layout).

**The throughput imbalance that makes softmax expensive.** Matmul and the special functions
needed by softmax live on different units with very different throughput. On the H100 SXM5,
FP16 matmul peaks near 989 TFLOPS, while the multi-function unit that computes the exponential
delivers only about 3.9 TFLOPS (16 special-function operations per SM per clock × 132 SMs ×
1.83 GHz) — a ratio of roughly 256x. For an FP16 forward pass at head dimension 128 there are
about 512x more matmul FLOPs than exponential FLOPs, but the exponential is 256x slower, so
the exponential can take on the order of half the wall-clock time of the matmul. Moving to
FP8 only sharpens this: matmul doubles, the exponential does not.

**The diagnostic finding that motivates everything.** The best existing fused exact-attention
kernel on Hopper measures at about 35% of peak FLOPs, against 80-90% for an optimized GEMM.
Part of this is using previous-generation (synchronous) Tensor Core instructions instead of
Hopper's; more fundamentally, that kernel adheres to a synchronous model — it issues a matmul,
waits, does softmax, waits, does the next matmul — so the asynchronous units it could overlap
instead idle in turn, and it never touches FP8. Separately, large language models are known to
carry *outlier features*: a small fraction of activation entries with magnitude far larger
than the rest (Dettmers et al. 2022; Sun et al. 2024), which is exactly the regime that makes
naive low-precision quantization (one scale factor per tensor) lose accuracy, since one
outlier inflates the scale and crushes the precision of every ordinary value.

## Baselines

These are the prior methods a new kernel would be measured against and would build on.

**Online (safe) softmax (Milakov & Gimelshein 2018).** The numerically stable softmax of a
vector needs the max and the sum of exponentials, ordinarily two passes over the data. The
online version computes both in a single streaming pass by carrying a running max `m` and a
running normalizer `d` and correcting `d` whenever the max grows:
```
m_0 = -inf, d_0 = 0
for j = 1..V:
    m_j = max(m_{j-1}, x_j)
    d_j = d_{j-1} · e^{m_{j-1} − m_j} + e^{x_j − m_j}
then  y_i = e^{x_i − m_V} / d_V.
```
A short induction proves `m_V = max_k x_k` and `d_V = Σ_j e^{x_j − m_V}`, so this equals the
two-pass safe softmax exactly. The combine step `[m_i, d_i] ⊕ [m_j, d_j] =
[max(m_i,m_j), d_i e^{m_i−max} + d_j e^{m_j−max}]` is associative and commutative, so the
reduction can be done block-wise in any order. **Gap:** this is a softmax primitive in
isolation; used directly on attention scores it still requires the full `S` to exist, so it
does not by itself avoid materializing the `N×N` matrices.

**Standard attention.** Compute `S = QK^T` and write it to HBM; read it back, compute
`P = softmax(S)`, write `P`; read `P` and `V`, compute `O = PV`, write `O`. This materializes
two `N×N` matrices, costing `O(N^2)` memory and `Θ(Nd + N^2)` HBM accesses; since softmax,
masking and dropout on the `N×N` matrix are memory-bound, the HBM traffic dominates the
runtime. **Gap:** the wall-clock time is set by quadratic HBM traffic, not by the
matmul-dominated FLOPs, and the intermediate matrices make long sequences infeasible.

**Tiled IO-aware fused attention (FlashAttention, Dao et al. 2022).** Split `Q, K, V` into
blocks, stream them through SMEM, and apply the online-softmax recurrence block-wise so the
output of each key/value block is folded into a running `(O, m, ℓ)` with the right rescaling —
the entire attention computation fuses into a single kernel that never writes `S` or `P` to
HBM. For the backward pass, store only `O` and the softmax statistics and *recompute* `S, P`
on chip from the blocks (selective checkpointing). The IO complexity drops from `Θ(Nd + N^2)`
to `Θ(N^2 d^2 / M)` for SRAM size `M`; since `d^2 ≪ M` for typical `d` (64-128) and `M`
(~100 KB), that is many times fewer HBM accesses, and a matching lower bound shows no exact
algorithm can asymptotically beat `Θ(N^2 d^2 / M)` across the SRAM range. **Gap:** it
parallelizes only over `(batch × heads)`, so occupancy collapses for long sequences with
small batch; its inner loop and "split-K" warp partitioning force intermediate results
through SMEM with synchronization; and, written for the previous synchronous model, it
under-utilizes compute on a newer chip.

**Better-parallelized fused attention (FlashAttention-2, Dao 2023).** Several refinements of
the tiled kernel that target the cost of the *non-matmul* work — important because on the
prior-generation A100 each non-matmul FLOP costs about 16x a matmul FLOP (312 vs 19.5
TFLOPs/s). (1) Defer the normalization: instead of dividing the running output by `ℓ` at every
block, carry an un-normalized accumulator `Õ` and divide once at the very end —
`Õ^{(2)} = diag(e^{m^{(1)}−m^{(2)}}) Õ^{(1)} + e^{S^{(2)}−m^{(2)}} V^{(2)}`, then
`O = diag(ℓ^{(last)})^{-1} Õ^{(last)}`. (2) Keep only the logsumexp `L = m + log(ℓ)` for the
backward pass rather than both `m` and `ℓ`. (3) Reorder the loops so the outer loop is over
query blocks, and parallelize across the sequence-length dimension (the loop-swap and
seq-length parallelism first implemented in the Triton fused-attention kernel), which restores
occupancy when batch×heads is small. (4) Partition the query block across the warps of a CTA
(rather than splitting K/V), so each warp produces a complete slice of the output with no
inter-warp reduction, cutting SMEM traffic. (5) For causal masking, skip blocks that are
entirely in the future, and apply the elementwise mask only on the diagonal-crossing boundary
blocks. **Gaps:** block sizes are tuned by hand over a few choices, with auto-tuning
left as future work; and the algorithm still follows a synchronous model — it makes no use of
the new chip's asynchronous Tensor Core / memory units, and runs only in 16-bit precision —
which is why it leaves the bulk of the new chip's peak FLOPs on the table.

## Evaluation settings

The natural yardsticks for a fused attention kernel, all pre-existing:

- **Hardware:** a single NVIDIA H100 SXM5 (80 GB) GPU.
- **Workloads:** causal self-attention forward pass across configurations spanning the regimes
  that stress different parts of the kernel, with total tokens held fixed (e.g. 16384) so the
  batch shrinks as the sequence grows: head dimension 64 at sequence length ~4K (many heads,
  small batch), head dimension 128 at ~8K, head dimension 256 at ~16K (few heads, long
  context). FP16 inputs, causal masking.
- **Metrics:** achieved throughput in TFLOPs/s (primary; higher is better), kernel latency in
  milliseconds (lower is better), and a hard correctness gate — maximum absolute difference
  from a high-precision reference (e.g. a reference scaled-dot-product-attention implementation)
  must stay below a small threshold (e.g. `1e-2`). For accuracy studies, the root-mean-square
  error against an FP32-reference output, including a setting with deliberately injected
  outlier features in `Q, K, V`.
- **FLOP convention:** for the causal forward pass, count
  `4 · batch · heads · N^2 · d / 2` (the `1/2` for the masked-out upper triangle).
- **Protocol:** warm up the kernel, then time the median over many repetitions with CUDA
  events; check correctness before timing.

## Code framework

The kernel plugs into a fixed benchmark harness that already exists: it generates `Q, K, V`
of shape `(batch, heads, seqlen, headdim)` in FP16, calls a reference attention for a
correctness check, then times the candidate. The candidate is a single Python entry point
`custom_attention_forward(q, k, v, causal, sm_scale)` that may call GPU kernels written in a
tile-based GPU DSL (for example, `@triton.jit` kernels, tile matmul, tiled HBM access,
elementwise tile operations, reductions, and compile-time configuration search). The
primitives that already exist are the DSL itself and the elementwise/reduction/dot operations
on tiles; everything specific to the candidate kernel is the open slot.

```python
import math
import torch
def custom_attention_forward(q, k, v, causal=True, sm_scale=None):
    """Return the exact row-wise softmax(QK^T)V output for tensors of shape
    (batch, heads, seqlen, headdim), matching the reference to < 1e-2."""
    batch, nheads, seqlen, headdim = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(headdim)
    # TODO: the implementation we will design.
    pass
```
