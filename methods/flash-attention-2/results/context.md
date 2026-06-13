# Context: the ground a faster exact-attention kernel stands on

## Research question

Self-attention is the runtime and memory bottleneck of the Transformer, and both scale quadratically in the sequence length `N`. Pushing context length well past the historical ~2k limit — to read books, high-resolution images, long videos — runs straight into this wall. The question is not how to *approximate* attention more cheaply (sparsity and low-rank methods exist but most large-scale runs still use exact attention), but how to compute *exact* attention as fast as the hardware will allow.

There is already an exact, IO-aware attention kernel that reorders the computation with tiling and recomputation to cut memory traffic, turning memory from quadratic to linear in `N` and giving a 2–4× wall-clock speedup with no approximation. But it still leaves most of the GPU on the floor: its forward pass reaches only ~30–50% of the device's theoretical FLOPs/s and its backward pass ~25–35%, whereas a well-tuned matrix multiply reaches 80–90%. So the precise problem is: take an already IO-optimal attention kernel and close the gap to GEMM-level throughput. Profiling points at two culprits — suboptimal work partitioning *between thread blocks* (low occupancy, idle SMs) and *between warps within a block* (unnecessary shared-memory traffic) — plus time wasted on operations the GPU is bad at. A solution must keep the exact output and the linear memory, while restructuring the parallelism and the per-step arithmetic so the kernel spends its time the way a fast matmul does.

## Background

**GPU performance model.** A GPU has a memory hierarchy — large, slow high-bandwidth memory (HBM: 40–80 GB at ~1.5–2.0 TB/s on an A100) and small, fast on-chip SRAM / shared memory (~192 KB per streaming multiprocessor, ~19 TB/s, across 108 SMs). SRAM is roughly an order of magnitude faster than HBM but tiny. Computation is organized as a *kernel* run by a massive number of *threads*; threads are grouped into *warps* of 32 (which can share data cheaply by register shuffles), warps into *thread blocks* (which share through shared memory, requiring writes/reads and synchronization), and thread blocks are scheduled onto SMs. Two facts drive everything that follows. First, *occupancy*: the GPU is only well-used when there are many thread blocks (say ≥80 on an A100) to keep all SMs busy. Second, *matmul vs non-matmul asymmetry*: modern GPUs have specialized matrix-multiply units (Tensor Cores) — an A100 does 312 TFLOPs/s of FP16/BF16 matmul but only 19.5 TFLOPs/s of non-matmul FP32, so each non-matmul FLOP is about 16× more expensive than a matmul FLOP. Even a small fraction of non-matmul work can dominate wall-clock time.

**Online (streaming) softmax.** The softmax over a row couples all entries through the max and the sum, which seemingly forces materializing the whole row. The online-softmax trick (Milakov & Gimelshein 2018; applied to attention by Rabe & Staats 2021) computes softmax block by block while maintaining a running max `m` and running denominator `ℓ`, rescaling the partial result when a new block raises the max — yielding the exact softmax with no approximation. For two column blocks `S^(1), S^(2)` of one row block, the running statistics update as `m^(2) = max(m^(1), rowmax(S^(2)))`, `ℓ^(2) = e^{m^(1)-m^(2)} ℓ^(1) + rowsum(e^{S^(2)-m^(2)})`, and the partial output is rebased to the new max before adding the new block's contribution.

**Attention itself.** Given `Q, K, V ∈ ℝ^{N×d}`, attention is `S = QKᵀ ∈ ℝ^{N×N}`, `P = softmax(S)` (row-wise), `O = PV ∈ ℝ^{N×d}`. The backward pass is `dV = PᵀdO`, `dP = dO Vᵀ`, `dS = dsoftmax(dP)`, `dQ = dS K`, `dK = Qᵀ dSᵀ`; the softmax gradient for one row is `ds = (diag(p) - ppᵀ)dp`. The Transformer (Vaswani et al. 2017) is the architecture this serves.

## Baselines

**Standard attention implementation.** Calls a GEMM for `S = QKᵀ` and writes the `N×N` matrix to HBM; loads it back to compute the softmax `P`, writes `P` to HBM; calls a GEMM for `O = PV`. Materializing `S` and `P` costs `O(N²)` memory, and `P` must also be kept for the backward pass. Since `N ≫ d` (typically `N` is 1k–8k, `d` is 64–128), this is dominated by HBM traffic and is memory-bandwidth-bound; the large number of HBM reads/writes makes it slow. Gap: quadratic memory and far more HBM traffic than necessary.

**IO-aware tiled attention (FlashAttention; Dao et al. 2022).** Keeps the exact output but never materializes `S` or `P` in HBM. Forward: *tiling* — load blocks of `Q, K, V` from HBM into SRAM, compute the attention for that block, and use *online softmax* to update a running output, rescaling as the running max grows. Backward: *recomputation* — recompute `S, P` from the SRAM-resident input blocks rather than storing them, so memory is linear in `N` (10–20× saving) instead of quadratic. Result: 2–4× faster than standard attention, no approximation. Gaps (the openings this work attacks): (1) it parallelizes only over the batch and head dimensions — one thread block per attention head — so when sequences are long (and batches consequently small) there are too few thread blocks to fill the SMs, hurting occupancy; (2) inside a block it uses a "split-K" warp partition (`K, V` split across the four warps, `Q` shared), which forces every warp to write partial results to shared memory and synchronize to reduce them; (3) its online-softmax bookkeeping rescales the output accumulator on *every* inner step, spending non-matmul FLOPs that, at 16× the cost, eat into throughput. The forward pass reaches only 30–50% of peak, the backward 25–35%.

**Approximate-attention methods (Reformer, Linformer, Performer, Longformer, BigBird, …).** Reduce the asymptotic cost by sparsifying or low-rank/kernel-approximating the attention matrix. Gap: they change the output (approximation) and, as the field has found, most large-scale training still uses exact attention — so they do not address the goal of a faster *exact* kernel.

## Evaluation settings

The natural yardstick is the attention kernel's wall-clock throughput, reported as a fraction of the device's theoretical maximum FLOPs/s (a fast GEMM hits 80–90%), measured on an A100 GPU across head dimensions (`d` = 64, 128) and with/without a causal mask, at the sequence lengths of interest (512 up to 16k+). Peak memory during training is tracked across sequence lengths (the IO-aware baseline already makes this linear in `N`). End-to-end, the protocol is training-throughput (TFLOPs/s per GPU and model-FLOPs-utilization) when the kernel is dropped into GPT-style Transformer training. All of these are pre-existing, kernel-agnostic measurements.

## Code framework

A custom attention primitive exposes a forward and a backward, each implemented as a *tiled* kernel: the inputs are partitioned into blocks small enough to live in SRAM, loops walk the block grid, and a parallel launch maps blocks of work onto the GPU's thread blocks and warps. The bodies below are `# TODO`.

```python
import torch

def attention_forward(Q, K, V, B_r, B_c):
    """Exact O = softmax(Q K^T) V, computed in SRAM-sized tiles, never materializing N×N.
       Returns O and whatever per-row statistics the backward pass needs."""
    N, d = Q.shape
    O = torch.zeros(N, d)
    # partition Q into row blocks of size B_r, K/V into column blocks of size B_c
    # TODO: implement the tiled forward pass.
    raise NotImplementedError

def attention_backward(Q, K, V, O, dO, stats, B_r, B_c):
    """Recompute S, P from input tiles (no stored N×N matrix) and produce dQ, dK, dV."""
    # TODO: implement the tiled backward pass.
    raise NotImplementedError

# Parallel launch (conceptual): the kernel is launched over a grid of
# thread blocks, and within each block the work is divided among warps.
def launch_grid(N, B_r, batch, heads):
    # TODO: define the launch.
    raise NotImplementedError
```
