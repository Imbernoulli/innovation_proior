# Context: preconditions for a faster exact-attention kernel

## Research Question

In a Transformer, self-attention runtime and memory grow with the sequence length `N`, because the score matrix has `N^2` entries. Long-document, code, image, audio, and video use cases push context length upward, and most large training and inference systems use exact softmax attention rather than a low-rank, sparse, or kernel approximation.

The question is a systems question: given an exact attention primitive that already avoids materializing the full `N x N` attention matrix, how close to GEMM-like throughput can the kernel run while keeping the mathematical output unchanged and the activation memory linear in `N`?

The constraint is that the target computation remains

```text
S = Q K^T
P = row_softmax(S)
O = P V
```

with optional scale, masking, and dropout around the score/probability matrices. The implementation may change the order of loads, reductions, and launches, but the returned `O` and the backward gradients must match exact attention up to normal floating-point roundoff.

## Hardware And Softmax Facts

An A100-class GPU has large high-bandwidth memory (HBM) and much smaller on-chip SRAM/shared memory per streaming multiprocessor (SM). HBM is large but slow relative to SRAM; SRAM is fast but only large enough for tiles. GPU work is launched as thread blocks scheduled on SMs, with threads grouped into warps of 32. Warps communicate cheaply within a warp, while communication across warps in a thread block generally goes through shared memory and synchronization.

The arithmetic units are also asymmetric. Tensor Cores make FP16/BF16 matrix multiply extremely fast, while scalar and reduction-style non-matmul work is far slower. On A100, the paper's hardware model uses 312 TFLOPs/s for FP16/BF16 matmul and 19.5 TFLOPs/s for non-matmul FP32, so a non-matmul FLOP costs roughly 16x a matmul FLOP. Once HBM traffic has been reduced, elementwise rescaling, exponentials, reductions, masking, and bounds checks become worth counting carefully.

The softmax obstruction is global by row. For a row split into column blocks, keep a running row max `m` and denominator `ell`. When a new block with scores `S_j` arrives,

```text
m_new = max(m_old, rowmax(S_j))
P_tilde_j = exp(S_j - m_new[:, None])
ell_new = exp(m_old - m_new) * ell_old + rowsum(P_tilde_j)
```

The factor `exp(m_old - m_new)` is always at most one, because the running max can only increase. It rebases the earlier exponentials from the old max to the new max. This is the core online-softmax identity that lets an exact softmax row be processed in blocks.

For the backward pass, if `dO` is the gradient of `O`, then

```text
dV = P^T dO
dP = dO V^T
dS = P * (dP - D[:, None])
D_i = sum_k dO_ik * O_ik
dQ = dS K
dK = dS^T Q
```

The row vector `D` is the row-wise dot product of `dO` and `O`; it is the same as `sum_j P_ij dP_ij`.

## Existing Baselines

The standard implementation calls a GEMM for `QK^T`, writes the `N x N` score matrix to HBM, reads it back for softmax, writes the `N x N` probability matrix, then reads that matrix again for `PV`. It also stores the probability matrix for the backward pass. This has quadratic memory in `N`.

The IO-aware exact baseline uses tiling and recomputation. It loads blocks of `Q`, `K`, and `V` into SRAM, computes a score tile, uses online softmax to update the output, and never stores the full `S` or `P` matrix in HBM. In the backward pass it recomputes score/probability tiles from `Q`, `K`, `V`, and saved per-row softmax statistics. This gives exact attention, `O(N^2 d)` arithmetic, and linear extra memory in `N`. It launches independent thread blocks over batch and heads, splits work inside a block over the key/value dimension, and performs an elementwise normalization in each online-softmax update.

Approximate-attention methods such as low-rank, sparse, locality-sensitive hashing, or kernelized attention reduce asymptotic cost by changing the attention operator. They are comparison points for a different operator.

## Evaluation Frame

The natural kernel-level measurements are forward and backward throughput as a fraction of theoretical device FLOPs/s, peak memory as sequence length grows, and latency across causal and non-causal attention at head dimensions such as 64 and 128. A strong GEMM can reach roughly 80-90% of peak.

The natural model-level measurements are training throughput and model FLOPs utilization when the attention primitive is inserted into GPT-style Transformer training. The implementation should preserve exactness, support causal masking, and keep memory linear in the sequence length apart from inputs and outputs.

## Starting Scaffold

The scaffold is a fused attention primitive with tiled forward and backward kernels. The unknowns are the loop order, the per-row softmax state, the launch grid, the block sizes, and the intra-block warp partitioning.

```python
import torch

def attention_forward(Q, K, V, B_r, B_c, *, causal=False, softmax_scale=1.0):
    """Exact O = softmax(softmax_scale * Q K^T) V in SRAM-sized tiles.
    The implementation must not materialize the full N x N score/probability matrix.
    It should return O plus the row statistics needed by the backward pass.
    """
    N, d = Q.shape
    O = torch.empty_like(Q)
    # TODO: choose the row/column tile loop, online-softmax state, and launch grid.
    raise NotImplementedError

def attention_backward(Q, K, V, O, dO, stats, B_r, B_c, *, causal=False, softmax_scale=1.0):
    """Recompute tiled scores/probabilities and return dQ, dK, dV."""
    # TODO: choose the backward parallelization and accumulation strategy.
    raise NotImplementedError

def launch_grid(N, B_r, batch, heads):
    """Conceptual GPU launch shape for the forward kernel."""
    # TODO: map independent tiles to thread blocks and map work inside a block to warps.
    raise NotImplementedError
```
