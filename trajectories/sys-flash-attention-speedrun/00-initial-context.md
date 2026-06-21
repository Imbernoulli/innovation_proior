## Research question

Compute **exact** scaled-dot-product attention — `out = softmax(QKᵀ / √d) · V`, bit-for-bit identical to the textbook formula — with memory that grows **linearly** in sequence length, as fast as possible on NVIDIA GPUs. The model, the math, and the output are fixed; only the kernel implementation is open. The yardstick is achieved **TFLOPs/s** and **Model FLOPs Utilization (MFU)**: how close the attention computation gets to the GPU's peak tensor-core throughput at a given sequence length and head dimension.

Attention is the per-layer bottleneck of every transformer. The two GEMMs in `QKᵀ` and `P·V` are dense tensor-core work; the softmax is a memory-bound reduction; and the naive recipe materializes an `N×N` score matrix that is quadratic in memory and HBM traffic.

## Prior art / Background / Baselines

A single attention head, the textbook way: form `S = QKᵀ / √d`, take row-wise softmax `P = softmax(S)`, then `O = P·V`. The two matmuls are `O(N²·d)` FLOPs each; the softmax is `O(N²)` elementwise work. On a GPU the tensor cores can run the GEMMs near peak. The standard recipe writes the `N×N` score matrix to HBM, reads it back for softmax, and writes the resulting probability matrix back to HBM before the second GEMM. For `N=8K` and FP16 that is 128 MB per head of intermediate that exists only to be consumed by the next step.

Existing directions:

- **Three-pass dense attention (e.g. `torch.nn.functional`-style).** A batched GEMM computes `S = QKᵀ`, a softmax kernel normalizes over the last dimension, and a second batched GEMM computes `O = P·V`. Each step is individually near-peak, and the full `N×N` `S` and `P` are written to and read from HBM between steps. Memory and HBM traffic scale as `O(N²)`.

- **Approximate / sub-quadratic attention (Reformer, Performer, Linformer, sparse attention, etc.).** These change the computation itself: kernel feature maps that linearize the softmax, locality-sensitive hashing, fixed or learned sparsity patterns, and low-rank projections of K and V. They reduce asymptotic FLOPs and memory, and compute a different, approximate attention.

- **Fused attention kernels.** A single CUDA kernel computes the score, softmax, and output matmuls without writing intermediate matrices back to HBM. Existing implementations cover specific head dimensions, inference-only forward passes, and non-causal or fixed-length masks.

- **Vendor GEMM and fused-attention libraries.** cuBLAS / cuBLASLt give near-peak batched GEMMs, and cuDNN is beginning to ship a fused multi-head-attention primitive. A library call removes the burden of writing a tensor-core GEMM by hand, while a generic batched GEMM materializes its output and a fused primitive exposes a fixed interface.

- **Mixed and low precision.** Running the matmuls in FP16/BF16 to use the tensor cores (with FP32 accumulation) is the established way to roughly double GEMM throughput versus FP32; FP8 on Hopper promises another doubling.

## Fixed substrate / Code framework

The frozen substrate is the exact attention definition and its reference check. The math is fixed: given `Q, K, V` of shape `(batch, seqlen, nheads, head_dim)`, the output must be `softmax(QKᵀ/√d + mask)·V` to floating-point tolerance. The Python entry point below is the fixed contract; the kernel behind it is the only editable surface.

```python
# Fixed contract: exact scaled-dot-product attention, linear in seqlen memory.
# The MATH is frozen and checked against a reference; the KERNEL behind this call is editable.

def attention_forward(q, k, v, softmax_scale, causal):
    # q, k, v: (batch, seqlen, nheads, head_dim)
    # returns out: (batch, seqlen, nheads, head_dim) == softmax(q @ k.transpose / sqrt(d) + mask) @ v
    # and the log-sum-exp per row needed by the backward pass.
    ...


def attention_backward(dout, q, k, v, out, lse, softmax_scale, causal):
    # gradients dq, dk, dv of the exact attention output.
    ...
```

## Editable interface

The editable surface is the CUDA kernel that implements `attention_forward` and `attention_backward`. Choices that are open include: how output rows are partitioned across threadblocks and warps; the tile shapes for Q, K, V, and the accumulator tiles; whether loads and matmuls are synchronous or asynchronous; the precision path (FP16/BF16/FP8) and accumulation type; how causal masking is applied; and what is recomputed versus stored for the backward pass. The CUDA stub below marks the space.

```cuda
// One CUDA threadblock's view of the computation for its slice of the output.
// Tile shapes, loop structure, load/matmul scheduling, precision, masking, and
// intra-block work partitioning are all editable.

template <typename Kernel_traits>
__global__ void attention_fwd_kernel(Params params) {
    // Each block is responsible for some portion of the (batch, head, query-rows) output.
    // It must produce, for its query rows, softmax(Q Kᵀ / √d) · V.
    //
    // TODO: which portion of the output does one block own, and how is the grid laid out over
    //       (batch, heads, query blocks)?
    // TODO: the inner loop over key/value blocks — what is kept resident on chip, and how is
    //       the row-wise softmax computed when the full score matrix is not materialized?
    // TODO: how do the warps/threads within the block split the two matmuls and the softmax?
}
```

## Evaluation settings

Throughput at fixed, exact output on a fixed GPU. Primary metrics: achieved **TFLOPs/s** (standard `≈4·N²·d` per head for the two matmuls, times batch and heads, with causal roughly halved) and **MFU** (achieved ÷ peak tensor-core TFLOPs/s for the precision). Correctness is the hard gate: the kernel must match the reference exact-attention implementation to floating-point tolerance and must use memory linear in seqlen. Runs are reported forward, backward, and combined, swept over sequence length (e.g. 512 to 16K), head dimension (64, 128, 256), and causal vs non-causal, batched to fill the GPU. Reference hardware is the A100 80GB SXM (peak BF16 ≈ 312 TFLOPs/s) for Ampere-era numbers; Hopper-era numbers are reported where H100 hardware is available.
