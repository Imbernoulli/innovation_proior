## Research question

Compute **exact** scaled-dot-product attention — `out = softmax(QKᵀ / √d) · V`, bit-for-bit the same numbers
the textbook three-pass formula produces, with memory that grows **linearly** in sequence length — as fast as
possible on NVIDIA GPUs. The model, the math, and the answer are all fixed: given Q, K, V of shape
`(batch, heads, seqlen, head_dim)`, the output is the softmax-weighted average of the value vectors, and any
implementation that changes that result (a low-rank approximation, a sparsity pattern, a kernel feature map) is
disqualified — the only thing being optimized is the *kernel*. The yardstick is achieved **TFLOPs/s** and
**Model FLOPs Utilization (MFU)** at a given seqlen and head dimension: how close the attention computation gets
to the GPU's peak tensor-core throughput.

This matters because attention is the per-layer bottleneck of every transformer, and its arithmetic is trivial
to state but brutal to run well. The FLOPs in `QKᵀ` and `P·V` are two matmuls the tensor cores can devour; the
softmax in between is a memory-bound reduction; and the naive way of stitching them together materializes an
`N×N` score matrix that is quadratic in both memory *and* HBM traffic. The gap between a correct attention
kernel and a *fast* one is the difference between attention dominating the wall-clock of a long-context model and
attention being nearly free — on identical silicon, with identical outputs.

## Background

A single attention head, the textbook way: form `S = QKᵀ` (an `N×N` matrix of scores, scaled by `1/√d`), take a
row-wise softmax `P = softmax(S)`, then `O = P·V`. The two matmuls are `O(N²·d)` FLOPs each; the softmax is
`O(N²)` elementwise work. On a GPU the FLOPs are not the problem — the tensor cores can do the two matmuls at
hundreds of TFLOPs/s. The problem is everything around them, and it is all knowable before a line of kernel is
written:

- **The score matrix is `O(N²)` and the naive implementation writes it to HBM — twice.** The standard
  three-kernel recipe (a GEMM that writes `S`, a softmax kernel that reads `S` and writes `P`, a GEMM that reads
  `P`) materializes the full `N×N` matrix in global memory and round-trips it. For `N=8K` and FP16 that is 128 MB
  *per head* of intermediate that exists only to be immediately consumed. The HBM read/write of this quadratic
  buffer, not the matmul FLOPs, is what dominates the runtime — attention is **memory-bound**, spending most of
  its time moving the score matrix to and from HBM rather than computing on the tensor cores.
- **GPU memory is a steep hierarchy and the fast tier is tiny.** Registers and on-chip SRAM (shared memory) are
  ~10–100× the bandwidth of HBM but measured in tens-to-hundreds of kilobytes per streaming multiprocessor; HBM
  is gigabytes but an order of magnitude slower. A kernel is fast when each byte it loads from HBM is reused many
  times out of SRAM/registers before being evicted, and slow when it streams data through SRAM once. The naive
  attention kernels do the latter: every element of `S` is written to HBM by one kernel and read back by the
  next.
- **Softmax needs a max for numerical stability, and the max is a global reduction over each row.** To avoid
  overflow, softmax is computed as `exp(xᵢ − max_j xⱼ) / Σ exp(xⱼ − max_j xⱼ)`. The `max` and the `sum` are
  reductions over the *entire* row of `N` scores — which is exactly why the textbook implementation needs the
  whole row of `S` resident before it can normalize, and why it materializes `S`.
- **Tensor cores want big matmuls; the softmax in the middle stalls them.** `QKᵀ` and `P·V` are dense GEMMs the
  tensor cores run near peak, but the softmax between them is a reduction on the CUDA cores / SFUs. If the kernel
  computes all of `S`, then does the softmax, then does `P·V`, the tensor cores sit idle during the softmax and
  the softmax units sit idle during the matmuls — the two kinds of work serialize instead of overlapping.
- **The backward pass naively needs `S` and `P` saved from the forward.** Gradients through attention depend on
  the softmax probabilities `P`; the obvious implementation stashes the `N×N` `P` (or `S`) from the forward pass
  to reuse in the backward, which is the same quadratic-memory cost again, now as a saved activation that lives
  across the whole forward-backward.
- **A single GPU is a generational moving target.** Ampere (A100) added BF16/TF16 tensor cores at ~312 TFLOPs/s
  and asynchronous global→shared copies (`cp.async`). Hopper (H100) went further: a dedicated **Tensor Memory
  Accelerator (TMA)** that bulk-copies tiles between HBM and SRAM as a single asynchronous instruction;
  **warpgroup-wide async matmul (WGMMA)** that issues a large tensor-core GEMM from shared memory without
  blocking the issuing threads; and an FP8 tensor-core path at roughly double the FP16 rate. A kernel tuned for
  the synchronous Ampere model leaves much of a Hopper chip idle, because it neither overlaps TMA loads with
  compute nor keeps the asynchronous matmul pipeline full.
- **Low precision trades accuracy for throughput, and attention has outliers.** FP16/BF16 already halve the bytes
  and double the tensor-core rate versus FP32; FP8 (e4m3) doubles it again. But FP8 has ~3 mantissa bits, and
  attention activations are known to have large-magnitude outlier entries that, quantized naively to FP8, blow up
  the error. Using the fastest precision without wrecking the numbers is a known hazard, not a free lunch.

## Baselines

The ladder climbs out of the standard ways attention was computed before a fused, memory-aware kernel existed,
and the library and approximation directions that were on the table.

- **The three-pass / `torch.nn.functional`-style dense attention.** Issue a batched GEMM for `S = QKᵀ`, a softmax
  kernel over the last dimension, and a second batched GEMM for `O = P·V`. Each step is itself near-peak
  (cuBLAS GEMMs, a tuned softmax), so the *FLOPs* are efficient; the cost is that the full `N×N` `S` and `P` are
  written to and read from HBM between the steps. Its limitation is structural: memory and HBM traffic scale as
  `O(N²)`, so at long context the kernel is bandwidth-bound and the activation footprint (and the saved-for-
  backward `P`) becomes the capacity ceiling — long sequences either run slowly or do not fit at all.
- **Approximate / sub-quadratic attention (the Reformer / Performer / Linformer / sparse-attention line).** A
  large body of work attacked the `O(N²)` cost by *changing the computation*: low-rank kernel feature maps that
  linearize the softmax, locality-sensitive hashing that attends only within buckets, fixed or learned sparsity
  patterns, low-rank projections of K and V. These reduce asymptotic FLOPs and memory, but they pay for it in
  quality — they compute a *different*, approximate attention — and in practice their wall-clock speedups on real
  sequence lengths were often modest because they trade dense tensor-core matmuls for scattered, low-intensity
  operations. The gap they leave open is the one this task is defined around: they do not compute *exact*
  attention, so any regression in model quality is on them, and the question of how fast *exact* attention can be
  made is untouched.
- **Hand-fused attention kernels and the streaming-softmax primitive.** It was understood that fusing `QKᵀ`,
  softmax, and `P·V` into one kernel would avoid the HBM round-trip of `S`, and that a softmax can be computed in
  a single streaming pass that maintains a running maximum and running normalizer as new scores arrive (the
  "online softmax" reduction), rather than the standard two passes over a fully-materialized row. These pieces
  existed in the literature and in scattered kernels; assembling them into a complete, correct, GPU-saturating
  attention kernel — forward and backward, all head dimensions, causal masking — was the open engineering work.
- **Vendor GEMM and fused-attention libraries.** cuBLAS / cuBLASLt give near-peak batched GEMMs for the two
  matmuls, and the cuDNN library was beginning to ship a fused multi-head-attention primitive. A library call
  removes the burden of writing a tensor-core GEMM by hand, but a *generic* batched GEMM still materializes its
  output, and a closed fused primitive is a fixed black box: it does not expose the tiling, the precision, the
  masking, or the work-partitioning choices that a custom kernel can tune per GPU generation.
- **Mixed and low precision.** Running the matmuls in FP16/BF16 to use the tensor cores (with FP32 accumulation)
  was the established way to roughly double GEMM throughput; FP8 on Hopper promised another doubling. The known
  failure modes are numerical: half precision narrows the dynamic range, and FP8's few mantissa bits combined
  with attention's outlier activations make naive quantization lose accuracy.

## Evaluation settings

The yardstick is throughput at fixed, exact output on a fixed GPU. Primary metrics: **TFLOPs/s** (achieved
attention FLOPs ÷ wall-clock; the attention FLOP count is the standard `≈4·N²·d` per head for the two matmuls,
times batch and heads, with the causal case roughly halved) and **MFU / hardware utilization** (achieved
TFLOPs/s ÷ the GPU's peak tensor-core TFLOPs/s for the precision in use; higher is better). Correctness is the
hard gate: the kernel's output must match a reference exact-attention implementation to floating-point tolerance
(the same `softmax(QKᵀ/√d)·V`), and memory must be linear in seqlen. Runs are reported as forward, backward, and
combined, swept over sequence length (e.g. 512 to 16K), head dimension (64, 128, 256), and the causal vs
non-causal mask, batched to fill the GPU. Reference hardware spans the generations the kernel targets: A100 80GB
SXM (peak BF16 ≈ 312 TFLOPs/s) for the Ampere-era numbers and H100 80GB SXM (peak FP16 ≈ 989 TFLOPs/s, FP8 ≈
1979 TFLOPs/s) for the Hopper-era numbers.

## Code framework

The substrate is a single attention kernel behind a fixed Python entry point: it takes Q, K, V and returns the
exact softmax-weighted output, and it must also produce a backward pass for training. The frozen, never-edited
part is the **math** — the definition `out = softmax(QKᵀ/√d)·V`, checked against a reference to floating-point
tolerance. The free slot is the **kernel**: how the two matmuls, the softmax, and the masking are mapped onto
the GPU's memory hierarchy and tensor cores, how the work is partitioned across the GPU's parallel units, and
which precision and which generation-specific instructions are used.

```python
# The fixed contract: exact scaled-dot-product attention, linear in seqlen memory.
# The MATH is frozen and checked against a reference; the KERNEL behind this call is the editable surface.

def attention_forward(q, k, v, softmax_scale, causal):
    # q, k, v: (batch, seqlen, nheads, head_dim)
    # returns out: (batch, seqlen, nheads, head_dim) == softmax(q @ k.transpose / sqrt(d) + mask) @ v
    # and the log-sum-exp per (row) needed by the backward pass.
    #
    # TODO: how are QK^T, the row-softmax, and (.)@V computed and laid out across the GPU's
    #       memory hierarchy and tensor cores? What never gets written to HBM, and what does?
    ...


def attention_backward(dout, q, k, v, out, lse, softmax_scale, causal):
    # gradients dq, dk, dv of the exact attention output.
    # TODO: what is recomputed vs. read back, given the forward's choices above?
    ...
```

```cuda
// One CUDA threadblock's view of the computation it must perform for its slice of the output.
// The TILE SHAPES, the LOOP STRUCTURE, the choice of synchronous vs asynchronous loads and matmuls,
// the precision, and how threads/warps within the block divide the work are all the editable surface.

template <typename Kernel_traits>
__global__ void attention_fwd_kernel(Params params) {
    // Each block is responsible for some portion of the (batch, head, query-rows) output.
    // It must produce, for its query rows, softmax(Q Kᵀ / √d) · V.
    //
    // TODO: which portion of the output does one block own, and how is the grid laid out over
    //       (batch, heads, query blocks)?
    // TODO: the inner loop over key/value blocks — what is kept resident on chip across it,
    //       and how is the running softmax maintained so the N×N matrix never materializes?
    // TODO: how do the warps/threads within the block split the two matmuls and the softmax?
}
```

The starting point that fills these stubs is the straightforward fused tile: load a block of Q, loop over blocks
of K and V computing partial scores, softmax, and partial outputs, and write the result — leaving open exactly
*how* the tiling, the streaming softmax, the work partition across warps, the asynchronous loads/matmuls, and
the precision are chosen. Each rung below replaces one of these implementation choices with a faster one while
leaving the computed result — the exact attention output — unchanged.
