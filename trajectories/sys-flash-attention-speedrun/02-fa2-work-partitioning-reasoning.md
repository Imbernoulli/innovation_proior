The fused kernel is exact and linear in memory, the score matrix never hits HBM, and attention is finally
compute-bound. But "compute-bound" doesn't mean "at peak" — on an A100 the kernel reaches only a fraction of the
312 TFLOPs/s the tensor cores can deliver. The matmul math is already on the tensor cores; the inefficiency has
to be in *how the work is laid out across the GPU*, not in the math. So let me look at the kernel as a scheduling
problem: how many parallel units are busy, how much of the work each does is matmul versus not, and whether the
threads inside a block are fighting each other. Three separate leaks, and they're all about partitioning, not
arithmetic.

First, the grid. How many threadblocks does the forward launch, and is that enough to fill the GPU? In the fused
design I parallelized over the query blocks and over `(batch × heads)` — one block per `(query-tile, batch, head)`.
That's fine when `batch × heads` is large. But the regime I actually care about for long context is *small* batch
and *long* sequence: `batch=1`, a handful of heads, `N=16K`. Then `batch × heads` is tiny, and even with the query
tiling the number of blocks is small relative to the ~108 SMs on an A100 — most of the chip is idle. The sequence
dimension is enormous and I'm barely using it as a source of parallelism. The fix is to make the grid explicitly
range over the sequence (query) blocks as a first-class dimension: launch `num_m_block = ceil(seqlen_q / BLOCK_M)`
blocks along one grid axis, and `batch` and `heads` along the others, so the total block count is
`num_m_block × batch × heads` and scales with sequence length. Now a single long sequence on one head still
generates many blocks and saturates the SMs. Different query blocks are completely independent — each owns a
disjoint set of output rows, streams over all the keys itself, runs its own online softmax — so there's no
communication between them and the parallelization is free. This is the one-line change with the biggest reach:
the grid becomes `dim3(num_m_block, batch, heads)` and the occupancy problem at small batch / long context
disappears.

Second leak: the non-matmul FLOPs inside the inner loop. The tensor cores do the two GEMMs, but in between, every
key block, the online softmax does a pile of work on the *other* execution units — the CUDA cores and the special
function units — and that work doesn't overlap with the tensor cores; it stalls them. In the streaming softmax as
I first wrote it, each iteration rescales the running output accumulator by `exp(m_old − m_new)` *and* maintains a
running normalizer `ℓ`, and the final output is `acc_o / ℓ`. Look at what the rescale-by-`α` of the *whole output
accumulator* costs: `acc_o` is `BLOCK_M × head_dim` — a big tile — and I multiply every element of it by the
per-row scalar `α` on *every* key block. That's `O(BLOCK_M · head_dim)` of non-matmul multiplies per iteration,
and the tensor cores wait through it. Can I do less of it?

The division by `ℓ` is the easy one: dividing `acc_o` by the row sum is a *linear* operation that commutes with
all the accumulation, so there's no reason to track and apply it per iteration — I can carry the *unnormalized*
output through the whole loop and divide by the final `ℓ` exactly once, in the epilogue. So the per-iteration
`1/ℓ` work vanishes; only one division per row survives, at the end. That's the deferred normalization: keep `ℓ`
as a running scalar, never touch `acc_o` with it until the loop is done.

The max-rescale is subtler because `m` genuinely changes block to block, so `acc_o` really does need correcting
when the max grows — I can't defer *that* the same way. But I can be careful that the rescale is the *only*
non-matmul touch of `acc_o`, and that I'm not redundantly recomputing exponentials or doing the normalizer update
with extra passes. Reorganizing so that each iteration does: one tensor-core GEMM for `S`, one max-reduction, one
`exp`, one rescale of `acc_o`, one tensor-core GEMM for `P·V` — and pushing every operation that *can* leave the
loop out of it — minimizes the non-matmul fraction. A useful micro-trick falls out of the `exp`: computing
`exp2(x · log₂e)` instead of `exp(x)` lets the hardware use the fast base-2 exponential on the SFU and lets the
compiler fold the scale into a fused multiply-add, so I precompute `softmax_scale · log₂e` once and the inner loop
exponentiates in base 2. Every non-matmul cycle I remove is a cycle the tensor cores aren't stalled. The point of
all this is the ratio: attention's useful work is the two matmuls, and anything else in the loop is overhead that
holds the tensor cores idle, so I drive that overhead toward the floor.

Third leak, and the one I'd never have found without thinking about the *warps* inside a block: how is the work
split among the 4 (or 8) warps of a threadblock? The fused kernel computes a `BLOCK_M × BLOCK_N` score tile, and
I have to decide which warp computes which part. The naive split — the one a generic GEMM scheduler reaches for —
is to partition the *K* (and *V*) dimension across warps: each warp handles a slice of the keys, computes partial
scores and partial `P·V` for its key-slice, and then the warps *combine* their partials. Let me trace what that
combination costs. For the first GEMM `S = QKᵀ`, splitting keys across warps means each warp gets a `BLOCK_M ×
(BLOCK_N/nwarps)` slab of `S` — fine. But then the softmax is a reduction *across the key dimension* (the row max
and row sum run over all keys), so warps holding different key-slices of the same query row must exchange their
partial maxima and partial sums to agree on the row's `m` and `ℓ`. That's a cross-warp reduction through shared
memory, with a `__syncthreads`, on *every* key block. And for the `P·V` GEMM, each warp produces a partial output
for its key-slice and the partials must be *summed* across warps — another shared-memory reduction and barrier.
This is the "split-K" pattern, and on this problem it's a tax: the warps spend time writing partials to shared
memory, synchronizing, and reading them back, every iteration.

Flip it. Split the *query* dimension across warps instead: each warp owns a disjoint slab of the `BLOCK_M` query
rows, and computes the *full* `K`-range for its own rows. Now a single warp holds an entire query row's scores
across all keys in the block (it sees the whole `BLOCK_N` of keys), so the row's softmax — max, exp, sum, the
online-softmax update, the `P·V` accumulation — is *entirely local to that warp*. No warp needs another warp's
partials for its rows; there's nothing to reduce across warps and no barrier in the inner loop. K and V are
loaded once into shared memory and *all* warps read the same K, V tiles (each against its own queries), so the
shared-memory K/V traffic isn't even duplicated. The cross-warp communication that split-K paid every iteration is
gone. This is the work-partitioning insight: divide attention across warps along the dimension the softmax
*doesn't* reduce over (queries), not the one it does (keys), so each warp's softmax is self-contained. The cost of
the old scheme showed up as shared-memory reductions and `__syncthreads` in the hot loop; the new scheme has
neither.

None of this changes the math. The grid still covers exactly the same `(query, batch, head)` outputs; deferring
the `1/ℓ` division to the epilogue is algebraically identical (division commutes with the linear accumulation);
splitting queries across warps just reassigns which warp computes which output row. The result is bit-for-bit the
same exact attention. What changes is utilization: the GPU is fuller (sequence-parallel grid), the tensor cores
stall less (fewer non-matmul FLOPs, deferred normalization, base-2 exp), and the warps stop synchronizing in the
hot loop (query-split instead of key-split). All three are partitioning decisions, and together they roughly
double the throughput of the same fused kernel.

Let me pin the three changes in code. The grid that makes the sequence a parallel dimension
(`csrc/flash_attn/src/flash_fwd_launch_template.h`):

```cpp
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    // ...
    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
```

The inner loop — one GEMM for `S`, the online-softmax *rescale-of-output* fused into `softmax_rescale_o`, one GEMM
for `P·V` — with the `1/ℓ` normalization deferred to a single call after the loop
(`csrc/flash_attn/src/flash_fwd_kernel.h`):

```cpp
        FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // online softmax: rescale the running output by exp(m_old - m_new), no per-iter 1/ℓ
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);
        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
        Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
        FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }
    // Epilogue: the SINGLE 1/ℓ normalization for the whole row, done once after the loop
    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);
```

The causal chain: the fused kernel was compute-bound but under-utilized, and the slack was all in partitioning.
Making the query/sequence blocks a first-class grid dimension fills the SMs even at batch 1 and long context;
deferring the softmax normalization out of the inner loop (and using base-2 `exp`) cuts the non-matmul FLOPs that
were stalling the tensor cores; and splitting the warps over queries rather than keys makes each warp's softmax
self-contained, deleting the cross-warp reductions and barriers from the hot loop. Same exact output, same linear
memory — roughly twice the throughput, now reaching a large fraction of the chip's tensor-core peak. What remains
is that on the newest GPUs the tensor-core matmul and the data movement and the softmax still take turns on the
same warps; the chip has dedicated asynchronous engines for loading and for matmul that this synchronous kernel
doesn't yet exploit.
