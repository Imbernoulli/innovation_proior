The fused kernel is exact and linear in memory, the score matrix never hits HBM, and attention is finally
compute-bound. The end-to-end wins were real — the fused kernel already turned in a 3× training speedup on GPT-2 at
sequence length 1K — but an end-to-end number folds the whole model together and hides how the *attention kernel
itself* is doing against the silicon. So let me measure the kernel in isolation, in achieved TFLOPs/s against the
A100's 312 TFLOPs/s BF16 peak, and the honest reading is that it lands at only a fraction of that. "Compute-bound"
does not mean "at peak": the matmul math is already on the tensor cores, so the slack has to be in *how the work is
laid out across the GPU*, not in the arithmetic. Let me look at the kernel as a scheduling problem — how many
parallel units are busy, how much of what each does is matmul versus not, and whether the threads inside a block
are fighting each other. Three separate leaks, and they're all about partitioning, not FLOPs.

First, the grid. How many threadblocks does the forward launch, and is that enough to fill the chip? In the shipped
fused kernel the launch grid ran over `(batch × heads)` — one threadblock per attention head — and each block swept
its query tiles internally, looping the `BLOCK_M` row-blocks below the grid. That mapping is fine when
`batch × heads` is large: on an A100 with 108 SMs, and wanting ~2 resident blocks per SM to hide latency, I want on
the order of 216-plus concurrent blocks and several waves beyond that to amortize the tail. But the regime I
actually care about for long context is *small* batch and *long* sequence: `batch=1`, a GPT-2-style 12 heads,
`N=16K`. Then `batch × heads = 12`. Twelve blocks on a 108-SM chip is `12/108 ≈ 11%` SM coverage — eighty-odd SMs
sit dark — and the enormous sequence dimension, which is exactly where the parallelism is, is trapped as a serial
loop *inside* each of those twelve blocks. I am hoarding the one axis that is huge.

The fix is to lift the query/sequence tiling out of the inner loop and make it a first-class *grid* dimension:
launch `num_m_block = ceil(seqlen_q / BLOCK_M)` blocks along one grid axis and `batch`, `heads` along the others, so
the grid is `dim3(num_m_block, batch, heads)` and the total block count scales with sequence length. Run the numbers
on the same hard case: `N=16K`, `BLOCK_M=128` gives `num_m_block=128`, so `128 × 1 × 12 = 1536` blocks — fourteen
waves over 108 SMs, every SM busy, the tail wave `1536 − 14·108 = 24` blocks (about 1.5% of the work) negligible.
Different query blocks are completely independent — each owns a disjoint set of output rows, streams over all the
keys itself, runs its own online softmax — so there is no communication between them and the parallelization is
free. This is the one-line change with the longest reach: promoting the sequence to a grid axis converts an 11%-full
chip into a saturated one at batch 1, and it costs nothing but the launch geometry.

A grid can be large on paper while only one block fits per SM, so "1536 blocks" only helps if two actually reside
per SM. The binding resources are registers and shared memory. With the query-split layout below, one threadblock
(4 warps, 128 threads) holds an `acc_o` of `128 × 128` FP32 = 16384 registers and a `128 × 64` FP32 score tile =
8192 registers, ~24576 plus loop temporaries — call it `~200` registers/thread × 128 ≈ 25.6K, so `65536 / 25600 ≈
2.5` allows 2 resident blocks per SM. Shared memory: the `Q` tile (32 KB) plus double-buffered `K`,`V` (2 × 32 KB)
≈ 96 KB against 164 KB also allows nearly two. So ~2 blocks/SM × 108 SMs ≈ 216 concurrent blocks — the fill target
— and the 1536 launched blocks queue as ~7 waves on top, oversubscribing the chip enough to hide load latency
between waves.

One alternative is worth pricing before I commit: split the *key/value* range across independent blocks too — each
handling a slab of keys for the same query tile, with a second reduction kernel merging their partial `(õ, ℓ, m)`
by the online-softmax combine. That manufactures parallelism along the key axis, and it is the right move when a
single query tile cannot fill the grid — tiny batch, few query blocks, huge context. But in the regime I am in —
batch 1, 12 heads, `N`=16K — I already launch 1536 query blocks against 216 resident, oversubscribed sevenfold
without touching the key axis; splitting keys on top would only add a kernel launch and an HBM round-trip of the
partials for no occupancy I lack. So it stays a tool for the decode/tiny-batch case, not this sweep; sequence-
parallelism dominates it here.

Second leak: the non-matmul FLOPs inside the inner loop. The tensor cores do the two GEMMs, but in between, every
key block, the online softmax does a pile of work on the *other* execution units — the CUDA cores and the special
function units — and that work does not overlap with the tensor cores; it stalls them. Let me count it against the
matmul to see whether it can possibly matter, because "softmax is cheap" is another slogan I want to price. Per
query row over the full key range `N` at head dim `d`: the two matmuls are `QKᵀ` at `2Nd` FLOPs and `P·V` at
`2Nd`, so `~4Nd` of tensor-core work per row. The softmax touches, per row: a running max (`~N` compares), the
`exp` (`~N`), the row-sum (`~N`), and — the expensive structural one — the rescale of the *whole* output
accumulator `acc_o` by `α=exp(m_old − m_new)` on every key block. `acc_o` is `BLOCK_M × d`, so per row that rescale
is `d` multiplies per key block × `N/BLOCK_N` blocks. With `d=128`, `BLOCK_N=64` that is `128·N/64 = 2N`. And if I
*also* divide by the running `ℓ` inside the loop — the naive per-iteration normalization — that is another `d` per
block, another `2N`. So the non-matmul bill is roughly `3N` (max/exp/sum) `+ 2N` (accumulator rescale) `+ 2N`
(per-iter `1/ℓ`) `≈ 7N` per row, against `4Nd = 512N` of matmul. That is `7/512 ≈ 1.4%` of the FLOPs.

One-and-a-half percent sounds ignorable until I put it on the right execution unit. The A100 does FP32/SFU work at
about `19.5` TFLOPs/s against the `312` TFLOPs/s tensor peak — a `312/19.5 = 16×` throughput gap. Since the
non-matmul work runs `16×` slower, its share of *time* is not `1.4%` but `16·0.014 / (16·0.014 + 0.986)
= 0.224/1.21 ≈ 18%`. So a 1.4%-of-FLOPs softmax eats nearly a *fifth* of the wall clock, and every one of those
cycles is one the tensor cores spend stalled. That is the leak, quantified — and it also tells me exactly what to
cut, because every non-matmul FLOP I remove is worth `16×` its weight.

Two removals, both algebraic and both leaving the output identical. The division by `ℓ` is the free one: dividing
`acc_o` by the row sum is *linear* and commutes with the whole accumulation, `(Σ_j c_j)/ℓ = Σ_j (c_j/ℓ)`, and `ℓ`
is just a per-row scalar constant with respect to the block sum. So there is no reason to apply it per iteration —
carry the *unnormalized* `acc_o` through the entire loop and divide by the final `ℓ` exactly once, in the epilogue.
That deletes the `2N` per-iter-`1/ℓ` term outright, dropping the softmax bill from `~7N` to `~5N` and its time share
from `~18%` toward `~13%`. This is literally the single final division I already wrote at the tail of the fused
forward — I am now insisting it stay the *only* one. The max-rescale I cannot defer the same way, because `m`
genuinely changes block to block and `acc_o` really is stale by `α` when the max grows; that `2N` is load-bearing.
But I can make sure it is the *only* non-matmul touch of `acc_o` left inside the loop and that nothing else is
recomputed. The second removal is a micro-trick on the `exp`: the SFU has a native base-2 exponential
(`MUFU.EX2`), and `exp(x) = 2^{x·log₂e}` exactly. If I
compute `exp2(x · log₂e)` instead of `exp(x)`, the hardware runs the fast path directly; and by folding
`log₂e` into the precomputed `softmax_scale · log₂e` I hand the base change and the `√d` scaling to the compiler as
one fused multiply-add on the scores, saving a multiply per score element rather than paying `exp` = `ex2` + a
separate multiply. The point throughout is the ratio: attention's useful work is the two matmuls; anything else
in the loop is overhead holding the tensor cores idle at a `16×` penalty, so I drive that overhead toward the floor.

Third leak, and the one I would never have found without thinking about the *warps* inside a block: how is the work
split among the 4 (or 8) warps of a threadblock? The fused kernel computes a `BLOCK_M × BLOCK_N` score tile, and I
have to decide which warp computes which part. The naive split — the one a generic GEMM scheduler reaches for — is
to partition the *K* (and *V*) dimension across warps: each warp handles a slice of the keys, computes partial
scores and partial `P·V` for its key-slice, and then the warps *combine* their partials. Let me trace what that
combination costs, because that is where the barriers hide. For the first GEMM `S = QKᵀ`, splitting keys across four
warps gives each warp a `BLOCK_M × (BLOCK_N/4)` slab of `S` — fine so far. But the softmax is a reduction *across
the key dimension* (the row max and row sum run over all keys), so the four warps holding different key-slices of
the same query row must exchange their partial maxima and partial sums to agree on that row's `m` and `ℓ`. That is a
cross-warp reduction: a `log₂4 = 2`-round tree through shared memory, gated by a `__syncthreads`, on *every* key
block. And for the `P·V` GEMM, each warp produces a partial output for its key-slice and those partials must be
*summed* across the four warps — another shared-memory reduction and another barrier. Count it: at `N=8K`,
`BLOCK_N=64`, a query tile iterates `N/BLOCK_N = 128` key blocks, so split-K pays ~128 softmax reductions and ~128
output reductions per query tile in the forward — some 256 barriers, each a smem write-read round-trip plus a
`__syncthreads` that idles all four warps until the slowest arrives. Against a hot loop whose useful work is two
small tensor-core GEMMs the cores retire in a few hundred cycles, a per-iteration barrier of tens-to-hundreds of
exposed cycles is not a rounding error — it is a comparable-order stall stacked on every iteration.

Flip it. Split the *query* dimension across warps instead: each of the four warps owns a disjoint `BLOCK_M/4 = 32`-row
slab of the query tile and computes the *full* `K`-range for its own rows. Now a single warp holds an entire query
row's scores across all `BLOCK_N` keys in the block, so that row's softmax — max, exp, sum, the online-softmax
update, the `P·V` accumulation — is *entirely local to the warp*, needing only warp-shuffle reductions and no shared
memory and no `__syncthreads`. No warp needs another warp's partials; there is nothing to reduce across warps and no
barrier in the inner loop. K and V are loaded once into shared memory and *all four* warps read the same K, V tiles
(each against its own queries), so the smem K/V traffic is not even duplicated. The `~128 + 128` block-level
barriers per query tile that split-K paid collapse to zero. This is the work-partitioning insight stated as a rule:
divide attention across warps along the dimension the softmax *does not* reduce over (queries), not the one it does
(keys), so each warp's softmax is self-contained.

The three attack disjoint losses, so they compound. In the small-batch long-context regime the grid change alone
lifts SM coverage from ~11% toward saturation — a several-fold occupancy gain by itself. On the now-full chip, a
single well-fed block still spent order-`0.18` of its time stalled on non-matmul softmax and a comparable slice on
hot-loop synchronization; driving both toward zero (defer `1/ℓ`, delete the split-K barriers) is roughly
`1/(1−0.3) ≈ 1.4×` on that block, and multiplied through the batch-1 occupancy recovery the product lands near
`2×`. Coarse, but it says the target is reachable from these three levers alone.

None of this changes the math, and I can say why for each piece. The grid still covers exactly the same
`(query, batch, head)` outputs — it is the same set, relaunched with the sequence as an axis. Deferring the `1/ℓ`
division to the epilogue is the linear-commutes-with-sum identity I checked above, exact. Splitting queries across
warps just reassigns which warp computes which output row; every row is still computed once, by exactly one warp,
over the full key range. So the result is bit-for-bit the same exact attention, still linear in memory. What changes
is utilization: the GPU is fuller (sequence-parallel grid takes batch-1 long-context from 11% SM coverage to
saturated), the tensor cores stall less (the ~18%-of-time non-matmul share cut by deferring `1/ℓ` and folding the
base-2 `exp`), and the warps stop synchronizing in the hot loop (query-split deletes the ~256 per-tile barriers of
split-K). All three are partitioning decisions, and together they roughly double the throughput of the same fused
kernel.

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

Each change predicts a *different* falsifiable signature. The grid change should lift `tflops_per_sec` and `mfu`
most at small batch / long sequence and least where `batch × heads` was already ample; the softmax-FLOP cuts have a
head-dimension signature (the non-matmul fraction scales as `~5/(4d)`, so deferred normalization and base-2 `exp`
help more at `d=64` than `d=128`); the warp-split is a pure hot-loop win with no shape dependence. A
`speedup_over_prev` much below `~2×` would say one of the three leaks was not costing what I counted.

What remains after all three is a property of the kernel itself: it is synchronous. Within each warp the score
matmul, the data movement, and the softmax still take turns, each waiting on the one before it — so on hardware
fast enough to expose that serialization, the ceiling stops being how the work is partitioned and becomes the fact
that these three kinds of work never run at the same time.
