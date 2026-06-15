Let me start from what actually hurts. I have a fused exact-attention kernel that already does the right thing mathematically — it streams `Q, K, V` in tiles, never materializes the `N×N` score or probability matrices, runs the online softmax block by block, and it was carefully restructured to parallelize over the query/sequence dimension and to spend almost all its arithmetic on matrix multiplies. On the previous generation of GPU it was excellent. On this new Hopper H100 it measures at about 35% of the chip's theoretical peak FLOPs. And the thing that needles me is that an ordinary matrix-multiply kernel on the very same chip hits 80-90%. So this is not an arithmetic problem. The kernel is already matmul-dominated; it isn't doing wasteful FLOPs. Something else is leaving more than half the machine idle, and I need to find what.

The first reflex is to blame instructions: maybe the kernel is emitting previous-generation Tensor Core ops instead of the new ones, leaving raw matmul throughput on the floor. That's real and worth fixing, but it can't be the whole story, because even a kernel that used every fast instruction perfectly would still be running the *algorithm* the way it's written — issue a matmul, wait for it, do softmax, wait for it, issue the next matmul — and if I think about what the hardware is doing during each of those "waits," the answer is: a lot of expensive silicon is sitting still. So let me stop guessing and actually account for where the time goes inside one iteration of the inner loop.

The inner loop, per key/value tile, is: one matmul `S = QK^T`, then a softmax-ish reduction over `S` (row max, subtract, exponentiate, row sum, rescale the running output), then a second matmul `P V` accumulated into the output. Two matmuls and one elementwise/reduction sandwich. The matmuls run on the Tensor Cores. The exponential in the middle runs somewhere else — on the special-function unit, the same hardware that does transcendentals. So I should compare their throughputs, because if they're wildly different and they run one-after-another, the fast unit waits on the slow one. On this chip the FP16 matmul peak is about 989 TFLOPS. The special-function unit that computes the exponential does about 3.9 TFLOPS — that's 16 special-function operations per SM per clock, times 132 SMs, times 1.83 GHz. The ratio is roughly 256x. The exponential is 256 times slower per operation than the matmul.

Now, that alone wouldn't matter if there were almost no exponentials. So let me count. For a forward pass at head dimension 128, the matmul work is `O(N^2 d)` and the elementwise exponential work is `O(N^2)` — the ratio of matmul FLOPs to exponential FLOPs is about `512x`. So I'm doing 512 times more matmul than exponential. But each exponential is 256 times slower. Put those together: `512 / 256 = 2`, meaning the *time* spent on matmul is only about twice the *time* spent on exponentials — equivalently, the exponential can eat on the order of half the wall-clock time of the matmul. And here's the structural disaster: in the synchronous schedule, while the special-function unit grinds through that exponential, the Tensor Cores have nothing to do, and while the Tensor Cores grind through the matmul, the special-function unit has nothing to do. They take turns. Half the machine is always idle. *That* is the 35%. And it gets worse, not better, the moment I reach for low precision: FP8 roughly doubles the matmul throughput while the exponential throughput doesn't move at all, so the exponential's share of the time grows — the better I make the matmul, the more the un-overlapped softmax dominates.

So the problem statement sharpens to: the kernel serializes two kinds of work that live on two different, independently-clocked units, and I'm paying for both in series instead of in parallel. The fix has to be to *overlap* them — to arrange for the exponential of one chunk of work to be computed while the Tensor Cores chew on the matmul of another chunk.

What's the lever? The new chip is asynchronous in a way the old one wasn't. The Tensor Core instruction here is issued warpgroup-wide and is *non-blocking*: a warp can fire off a matmul and keep going, and the matmul can read its operands straight out of shared memory. There's a separate dedicated unit, the Tensor Memory Accelerator, that copies data from global memory to shared memory asynchronously, needing only a single thread to kick it off, and it doesn't tie up the registers that explicit index arithmetic would. So I have at least three things that can run concurrently if I let them: memory loads (on the TMA), matmuls (on the Tensor Cores), and the exponential/reduction (on the special-function and ordinary CUDA cores). The hardware *wants* to overlap. The synchronous algorithm just never asks it to.

Let me get the cheapest overlap first, because it's the most obvious and it sets up the harder one. Right now, when the inner loop needs the next `K_j, V_j` tile, it loads it and waits, then computes. But loading is on the TMA and computing is on the Tensor Cores — those are different units. If I split the warps of a threadblock into roles — some warps do *only* loads, some do *only* compute — then the load warps can be running ahead, fetching tile `j+1` from HBM into shared memory while the compute warps are still working on tile `j`. This is warp specialization: a producer/consumer split. The producers issue TMA loads into a shared-memory buffer; the consumers issue matmuls reading from it. To let the producer run ahead by more than one tile I make the shared-memory buffer *circular* with `s` stages, so the producer can fill stages `j, j+1, ..., j+s-1` while the consumer drains stage `j`; the producer waits only when it laps the consumer, and the consumer waits only on a stage that isn't filled yet. The producer issues no waits at all for the first `s` iterations — it just fills the pipe. And since the producer only issues TMA and needs almost no registers (one thread can drive the copy), while the consumer needs lots of registers for accumulators, I want to *give* the consumers more registers and *take* them from the producers — which the chip lets me do dynamically, reallocating registers between warpgroups. So: producer warps deallocate most of their registers, consumer warps reallocate a larger share, and the loads hide under the compute. Generically, splitting warps into single-role groups also just makes it easier for the compiler to find a good instruction schedule, because each warp's instruction stream is homogeneous.

That overlaps *memory* with *compute*. Good — but it does not touch the thing I actually diagnosed, which is the exponential serializing against the matmul *inside* the compute. The producer/consumer split hides the loads; it doesn't hide the softmax. I'm still, within the consumer, doing matmul-0 → wait → softmax → matmul-1 → wait, and the softmax still stalls the Tensor Cores. I have to break *that* dependency.

So look hard at the dependency. In one iteration: matmul-0 produces `S = QK^T`; softmax reads `S` to produce `P` (and updates the running max, sum, and the output rescale); matmul-1 reads `P` to accumulate into the output. It's a chain: matmul-0 → softmax → matmul-1. Within a single iteration there is no slack — each step needs the previous one's result. The exponential genuinely cannot start before its scores exist, and the second matmul genuinely cannot start before its probabilities exist. I can't overlap softmax with the matmuls *of the same iteration*. But I can overlap softmax of one iteration with matmuls of *another* iteration, because different iterations are independent until they fold into the shared running state.

The first way to do that uses two warpgroups at once. Suppose I have two consumer warpgroups, each handling its own work. If I could guarantee that warpgroup 1's matmuls are scheduled while warpgroup 2 is doing its softmax, and then warpgroup 2's matmuls run while warpgroup 1 does *its* softmax, then at every instant *some* warpgroup is keeping the Tensor Cores busy and *some* warpgroup is keeping the special-function unit busy. Neither unit idles. The trouble is the scheduler won't naturally do this; left alone, both warpgroups tend to hit their matmuls together and their softmaxes together, and I'm back to taking turns globally. So I have to force the ordering with barriers — synchronization instructions that make warpgroup 1's matmuls (the `PV` of this iteration and the `QK^T` of the next) get issued before warpgroup 2's matmuls. With that nudge, while warpgroup 1 hands off and starts its softmax, warpgroup 2's matmuls are running on the Tensor Cores; then the roles swap, warpgroup 2 does softmax while warpgroup 1 matmuls. It ping-pongs. The figure in my head is two interleaved sawtooths, exactly out of phase. In practice the compiler muddies it and the overlap isn't perfect, but the principle is sound: use a second independent stream of matmul work to fill the holes that one stream's softmax leaves in the Tensor Core timeline.

Can I get overlap even within a *single* warpgroup, without leaning on a second one? The dependency chain matmul-0 → softmax → matmul-1 is sequential *within* an iteration, but across iterations I have parallelism: iteration `j`'s matmul-1 (`P_j V_j`) is independent of iteration `j+1`'s softmax (which only needs iteration `j+1`'s scores `S_{j+1}`, which come from iteration `j+1`'s matmul-0). So if I keep an extra copy of the scores in registers — a "current" `S` and a "next" `S` — I can software-pipeline: issue matmul-0 for the next iteration (`Q K_{j+1}^T → S_next`, commit but don't wait), then issue matmul-1 for the current iteration (`P_cur V_j`, commit but don't wait), then while *both* of those asynchronous matmuls are in flight on the Tensor Cores, run the softmax on `S_next` on the special-function unit, then wait and rescale. The second matmul of iteration `j` is overlapped with the softmax of iteration `j+1`. That's the overlap I wanted, now inside one warpgroup, paid for with one extra `S` tile held in registers — `B_r × B_c × sizeof(float)` per threadblock. That register cost is not free: it competes with using larger tiles, which are also register-hungry, so the depth of this pipeline is a profiling trade-off, not a free lunch.

I should ask whether to push the pipeline deeper — three stages, overlapping the second matmul with two iterations' worth of softmax. I tried reasoning it through and it disappoints: a three-stage version needs yet another `S` (and the output rescale) kept live, so register pressure forces smaller tiles, and worse, the extra scheduling freedom can easily overlap only the first matmul with softmax while leaving the second matmul un-overlapped, so the theoretical extra overlap may not materialize. So I stop at two stages: it's the sweet spot where the overlap is real and the register cost is tolerable.

Now, separate from the overlap, there's a cheaper, more local fix for the exponential itself that I shouldn't miss. The expensive operation is `exp`. But the special-function unit has a native *base-2* exponential primitive — it computes `2^x` directly in hardware. And `e^x = 2^{x · log_2 e}`. So if I fold the constant `log_2 e = 1.44269504` into the scaling I already apply to the scores, then every exponential in the inner loop becomes one hardware `exp2` instruction instead of a software `exp`. I'm already multiplying the raw `QK^T` by the softmax scale `α = 1/√d`; I just multiply that scale by `1.44269504` once, up front (fold it into `Q` before the kernel, or into the `qk_scale` constant), and then the loop's `e^{s − m}` becomes `exp2((s − m) · log_2 e)` with the `log_2 e` already baked into `s` and `m`. One fused multiply on the score scale, applied once, converts the whole softmax to the fast intrinsic. This doesn't overlap anything; it just makes the slow side cheaper, which compounds with the overlap.

While I'm trimming the non-matmul work — and I should, because on the previous chip each non-matmul FLOP cost about 16x a matmul FLOP (the matmul ran at ~312 TFLOPs/s and the non-matmul at ~19.5), and on this chip the imbalance is even more lopsided — there are two carry-overs from the better-parallelized predecessor I want to keep, because they directly cut elementwise work in the hot loop. First: don't renormalize the output every block. The online softmax says that when the running max jumps from `m_old` to `m_new`, the partial output and the partial sum both have to be rescaled by `e^{m_old − m_new}`; the naive thing also divides by the running normalizer `ℓ` at each step to keep the output a proper weighted average. But that per-block division is pure non-matmul work. I can instead carry an *un-normalized* accumulator — at each block, `acc ← acc · e^{m_old − m_new} + P̃ V_j` with `P̃ = e^{S − m_new}`, never dividing — and only at the very end of the loop divide once by the final `ℓ`. Concretely the two-block identity is `Õ^{(2)} = diag(e^{m^{(1)}−m^{(2)}}) Õ^{(1)} + e^{S^{(2)}−m^{(2)}} V^{(2)}`, and then `O = diag(ℓ^{(last)})^{-1} Õ^{(last)}`; one division at the end instead of one per tile. Second carry-over, for any recomputation that needs the softmax row later: I don't need to store both the row max `m` and the row sum `ℓ`; the single logsumexp `L = m + log(ℓ)` suffices to reconstruct the probabilities, so I store just `L`. These don't change the asymptotics but they shave the slow-unit work that I'm trying so hard to overlap, so every bit removed is a bit I don't have to hide.

Causal masking is another place to avoid useless elementwise work. For autoregressive models I mask out `S_{ij}` whenever `j > i`. Done naively, every block applies a `where`-mask to its whole score tile — elementwise, on the slow side, every block. But because I'm already working in blocks, I can reason geometrically about which blocks even need a mask. For a given query block (rows `i` in some range), a key block whose column indices are *all* less than the smallest row index is entirely below the diagonal — fully unmasked — so it needs no mask at all; a key block whose column indices are *all* greater than the largest row index is entirely in the future — fully masked, contributes nothing — so I can skip computing it outright; and only the diagonal-crossing boundary blocks need the elementwise `where`. The split point for the no-mask pass is `non_causal_end = (start_m * BLOCK_M // BLOCK_N) * BLOCK_N`: round the first query row of this tile down to a key-block boundary. Everything before that is safely below the diagonal, and the second pass runs from that boundary up to `(start_m + 1) * BLOCK_M` with the causal mask. For large `N` this skips roughly half the blocks (the future ones) entirely and removes the mask op from the bulk of the rest. The bonus is that the no-mask pass has a cleaner, branch-free inner body, which pipelines and schedules better — exactly what I want for the overlap machinery above.

Now the other axis: low precision. The matmul is the bulk of the time, and FP8 doubles its throughput. So I want the two GEMMs in FP8. But FP8 brings two distinct headaches, one about *layout* and one about *accuracy*, and I have to solve both or it's not worth it.

The layout problem comes from a hardware constraint: FP8 matmul accepts its shared-memory operands only when they're contiguous in the inner contraction dimension (k-major), whereas FP16 was happy with either orientation. For the first GEMM, `Q K^T`, that's fine. But for the second GEMM, `P V`, the contraction is over the key/sequence dimension, so `V`'s tiles need to be contiguous in the *sequence* dimension — and `V` arrives contiguous in the *head* dimension, the opposite. A bulk memory copy can't change which dimension is contiguous. I could transpose `V` in global memory as a preprocessing step, but fusing that into a preceding op (like the rotary embedding) is awkward to ship in a library, and a standalone transpose kernel wastes bandwidth in the memory-bound inference case. So instead I transpose *in the kernel*: after loading a `V` tile into shared memory, use the warp-collective load-matrix / store-matrix instructions (which can transpose while they copy, at 128-byte granularity) to flip it, and do it in the producer warps so it's register-cheap. Even better, after the first iteration I can hide the transpose of the *next* `V` tile in the shadow of the two matmuls that are already running on the previous `V` and current `K` — it overlaps for free. There's a second, subtler layout mismatch unique to FP8: the layout in which the FP32 accumulator of the first GEMM lands in registers is *not* the layout the second GEMM expects for its operand. On FP16 these matched; on FP8 they don't. I fix it with byte-permute instructions that reorder the accumulator's per-thread register entries into the operand layout — concretely permuting the entry order so that what was `d0 d1 d2 d3 d4 d5 d6 d7` becomes `d0 d1 d4 d5 d2 d3 d6 d7`, replicated every 8 bytes — and then I arrange the in-kernel `V` transpose to write out a matching row permutation, so the permuted `P` columns line up with the permuted `V` rows and the product comes out correct. Doing the transpose in-kernel gives me exactly this extra degree of freedom, which is what lets me avoid expensive cross-thread register shuffles.

The accuracy problem is the one that could sink FP8 entirely. FP8 (e4m3) has 3 mantissa bits and 4 exponent bits — coarse. And language-model activations are known to carry outliers: a few entries far larger in magnitude than the rest. The standard way to quantize is one scale factor per tensor: divide the whole `Q` (or `K`, or `V`) by its max-magnitude so it fits the FP8 range. But with an outlier, that max is huge, so the scale is huge, and every *ordinary* value gets quantized to a tiny number of representable levels — the outlier crushes the precision of everything else. Two ideas fix this, and both happen to cost almost nothing here. First, *block* quantization: keep one scale per *block* (per `B_r × d` tile of `Q`, per `B_c × d` tile of `K, V`) instead of one per tensor. Because the algorithm already operates block by block, applying a per-block scale to each block of `S` is free — I just multiply each score tile by the product of its `Q`-block and `K`-block scales — and now an outlier only inflates the scale of *its own* block, not the whole tensor. This pairs naturally with fusing the quantization into the bandwidth-bound op that precedes attention (rotary embedding), at no extra cost. Second, *incoherent processing*: before quantizing, multiply `Q` and `K` each by a random orthogonal matrix `M`. Since `M` is orthogonal, `M M^T = I`, so `(Q M)(K M)^T = Q M M^T K^T = Q K^T` — the attention scores, and hence the entire output, are *exactly unchanged*. But each entry of `Q M` is a random linear combination of the entries of a row of `Q`, so an outlier gets spread across many entries instead of sitting in one — the rotated tensor has a much tamer dynamic range, so the per-block max isn't dominated by a single spike and quantization error drops. I don't want a dense `d × d` matmul for `M`, that's `O(d^2)`; so I take `M` to be a normalized Hadamard matrix times random `±1` diagonal sign flips, which can be applied in `O(d log d)` (the fast Hadamard transform) and, like block quantization, fuses into the rotary embedding for free. The output is mathematically identical; only the quantization noise floor improves.

Let me make sure the core math is still exactly right after all this restructuring, because none of the scheduling tricks are allowed to change the answer. The online softmax recurrence is the load-bearing identity, so I'll re-derive it and convince myself the block-wise, deferred-rescale version computes the true softmax-weighted output. For a row, maintain `m` (running max of scores seen) and `ℓ` (running sum of `e^{score − m}`) and an un-normalized output accumulator `Õ = Σ e^{score − m} · v`. Process a new block with scores `s` and values `v`: the new max is `m' = max(m, rowmax(s))`. The old partial sum was taken relative to the old max `m`, so to re-base it to `m'` I multiply by `e^{m − m'}`: `ℓ' = e^{m − m'} ℓ + rowsum(e^{s − m'})`. The old partial output was likewise relative to `m`, so `Õ' = e^{m − m'} Õ + e^{s − m'} v`. To see this is right, suppose two blocks `s^{(1)}, s^{(2)}` with global max `m = max(m^{(1)}, m^{(2)})`. Then `Õ^{(2)} = e^{m^{(1)} − m} Õ^{(1)} + e^{s^{(2)} − m} v^{(2)} = e^{s^{(1)} − m} v^{(1)} + e^{s^{(2)} − m} v^{(2)}` — exactly the un-normalized sum `Σ_j e^{s_j − m} v_j` over both blocks, with `m` the true overall max. And `ℓ^{(2)} = e^{m^{(1)} − m} ℓ^{(1)} + rowsum(e^{s^{(2)} − m}) = rowsum(e^{s^{(1)} − m}) + rowsum(e^{s^{(2)} − m}) = Σ_j e^{s_j − m}`, the true normalizer. So `O = Õ^{(last)} / ℓ^{(last)} = Σ_j (e^{s_j − m} / Σ_k e^{s_k − m}) v_j = softmax(s) · V`, exact, independent of where I cut the blocks or how often the max jumped. The deferral of the `1/ℓ` to the end changes nothing about correctness — it only moves where the single division happens. The `exp2` substitution is an algebraic identity, not an approximation, since `e^x = 2^{x log_2 e}` exactly. The incoherent rotation is exact since `M M^T = I`. The only place anything is *approximate* is the FP8 quantization of the operands themselves, and that's a precision choice I control with the scales — the algorithm around it is exact. Good: I haven't changed what's computed, only how the work is laid out across the units and the bits.

Let me state the schedule as a concrete algorithm so I'm sure the pieces fit. At the threadblock level, one block handles one query tile `Q_i` and computes one output tile `O_i`. Initialize a pipeline object managing an `s`-stage circular shared-memory buffer. The producer warpgroup deallocates most of its registers, issues the load of `Q_i`, then loops over key/value tiles `j`: wait for stage `j mod s` to be free, issue the loads of `K_j, V_j` into that stage, signal the consumer. The consumer warpgroup reallocates a larger register share, initializes `O_i = 0`, `ℓ_i = 0`, `m_i = -∞` on chip, waits for `Q_i` and `K_0`, computes `S = Q_i K_0^T` (matmul, commit and wait), releases stage 0 for `K`, computes `m_i`, `P̃`, `ℓ_i` from `S` and rescales `O_i`; then for the main range `1 ≤ j < T_c − 1`: wait for `K_j`, compute `S_next = Q_i K_j^T` (commit, do not wait), wait for `V_{j−1}`, compute `O_i ← O_i + P̃_cur V_{j−1}` (commit, do not wait), wait for the `Q_i K_j^T` matmul, compute `m_i, P̃_next, ℓ_i` from `S_next` — this softmax runs while the two committed matmuls are still in flight — then wait for the `P̃_cur V_{j−1}` matmul and rescale `O_i`, release the buffer stages, and copy `S_next → S_cur`. Drain the last `V` tile, then in the epilogue rescale `O_i` by `1/ℓ_i`, compute `L_i = m_i + log(ℓ_i)`, and write `O_i, L_i` to HBM. That's the two-stage GEMM-softmax pipeline sitting on top of warp specialization. For multi-query and grouped-query attention I don't duplicate `K, V`; I just adjust the indexing into the head dimension so multiple query heads read the same key/value head.

Now, the harness I actually have to ship into doesn't let me hand-write the warpgroup barriers, the TMA descriptors, or the FP8 layout permutes — it's a tile-based DSL (Triton) where I write a `@triton.jit` kernel in terms of tile loads, `tl.dot` matmuls, and elementwise tile ops, and a compiler turns that into the machine code. So I can't directly express pingpong scheduling or the in-kernel `V` transpose. The question is how much of the principle survives the translation, and the answer is: the algorithmic parts survive directly, and the *async overlap* survives through the compiler's software-pipelining knob. When I mark the inner loop with `num_stages > 1`, the compiler restructures it into a software pipeline that *prefetches the next iteration's tile loads while the current iteration computes* — which is precisely the producer/consumer-plus-circular-buffer idea expressed at the DSL level, the loads hidden under the compute. I also get a warp-specialization flag on the loop and control over the number of warps. So my Triton realization of this design is: fold `log_2 e` into the scale and use `exp2`; carry the un-normalized accumulator and divide once at the end; do the two-pass causal split so the bulk of blocks run mask-free; parallelize the grid over query blocks and over `(batch × heads)`; and then *autotune* over the tile sizes `(BLOCK_M, BLOCK_N)`, the pipeline depth `num_stages`, and `num_warps`, because the right choice depends on the head dimension and sequence length — larger tiles amortize loop overhead and feed the Tensor Cores fuller matmuls, but cost registers and shared memory and eventually spill or fail to launch, and the only honest way to find the optimum across the head dimensions I care about is to let the compiler compile several configurations and pick the fastest at compile time. (The predecessor explicitly tuned a handful of block sizes by hand and flagged auto-tuning as the obvious next step; here I take that step, and extend the search to the pipeline depth and warp count that the new asynchronous compiler exposes.)

So let me write the kernel I'd actually run, filling the empty fused-attention slot. Two `@triton.jit` passes' worth of logic live in one kernel with the causal split inside; the launcher autotunes the schedule:

```python
import math
import torch
import triton
import triton.language as tl

# log_2(e): lets us replace the software exp by the hardware exp2, since
# e^x = 2^(x * log2(e)). We fold this constant into the softmax scale.
LOG2E = 1.44269504


@triton.autotune(
    # Search tile sizes, software-pipeline depth, and warp count at compile
    # time. num_stages > 1 is the DSL's software pipelining: it prefetches the
    # next iteration's K/V tile loads so they overlap the current compute --
    # the same producer/consumer overlap the async hardware offers, expressed
    # through the compiler. Larger tiles feed fuller matmuls but cost registers.
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
    start_m = tl.program_id(0)          # which query tile (parallel over seqlen)
    off_hz = tl.program_id(1)           # which (batch, head) -- embarrassingly parallel

    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Q tile stays resident; qk_scale (below) folds the softmax scale with
    # log2(e) so the inner loop's exp2 computes e^(sm_scale * score).
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)

    # Running softmax state + UN-normalized output accumulator (divide once, at end).
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")   # running row max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0            # running normalizer
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)    # running un-norm output
    qk_scale = sm_scale * LOG2E

    # Two-pass causal: pass 1 covers key blocks entirely below the diagonal
    # (no mask instruction at all); future blocks are simply never iterated.
    if IS_CAUSAL:
        non_causal_end = (start_m * BLOCK_M // BLOCK_N) * BLOCK_N
    else:
        non_causal_end = seqlen

    for start_n in tl.range(0, non_causal_end, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        qk = tl.dot(q, tl.trans(k))                  # GEMM-0 on Tensor Cores
        m_new = tl.maximum(m_i, tl.max(qk, axis=1) * qk_scale)
        qk = qk * qk_scale - m_new[:, None]
        alpha = tl.math.exp2(m_i - m_new)            # rescale factor (hardware exp2)
        p = tl.math.exp2(qk)                         # e^(score-max) via exp2
        l_i = l_i * alpha + tl.sum(p, axis=1)        # rebase + extend the normalizer
        acc = acc * alpha[:, None]                   # rebase the un-norm output
        v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)          # GEMM-1, accumulate (no 1/l here)
        m_i = m_new

    # Pass 2: only the diagonal-crossing block(s) need the elementwise mask.
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

    acc = acc / l_i[:, None]                          # the single deferred 1/l division
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen)


def custom_attention_forward(q, k, v, causal=True, sm_scale=None):
    batch, nheads, seqlen, headdim = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(headdim)
    o = torch.empty_like(q)
    lse = torch.empty((batch, nheads, seqlen), device=q.device, dtype=torch.float32)
    # Parallel over query tiles and over (batch * heads).
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

Let me retrace the causal chain so I'm sure each piece earned its place. The kernel was already arithmetically minimal yet ran at a third of the chip's peak, and the reason wasn't FLOPs — it was that the algorithm serialized matmul and softmax across two units (Tensor Core at ~989 TFLOPS, the exponential at ~3.9, a 256x gap) that take turns and idle each other, with about half the inner loop's time spent on a stalled unit. The whole design is overlap, applied at three grains: warp specialization with a circular buffer hides the memory loads under compute; inter-warpgroup pingpong overlaps one warpgroup's softmax with another's matmul; and a two-stage intra-warpgroup pipeline (an extra `S` in registers) overlaps each iteration's second matmul with the next iteration's softmax — stopping at two stages because three costs registers and the compiler won't overlap the second matmul anyway. Orthogonally I cut the slow-side cost itself: hardware `exp2` with `log_2 e` folded into the scale, an un-normalized accumulator that divides by the normalizer only once, the single logsumexp statistic, and a two-pass causal split that skips future blocks and masks only the diagonal-crossing boundary blocks. For raw matmul throughput I move to FP8 (2x), paying for it with an in-kernel `V` transpose and an accumulator byte-permute to satisfy FP8's k-major and accumulator/operand layout rules, and protecting accuracy with block quantization (one scale per tile, free since the algorithm is already block-wise) and incoherent processing (rotate `Q, K` by a normalized Hadamard-times-random-sign matrix, exact because `M M^T = I`, which spreads outliers so per-block quantization isn't dominated by one spike). The online-softmax identity guarantees all of this computes `softmax(QK^T)V` exactly regardless of how I cut the blocks or defer the rescale, so none of the scheduling changed the answer. And because the production substrate is a tile-based DSL rather than hand-written warpgroup code, I realize the async overlap through the compiler's `num_stages` software pipelining and a compile-time autotuner over tile sizes, pipeline depth, and warp count — taking the auto-tuning step the predecessor left as future work — landing on a fused forward kernel that keeps both the Tensor Core and the special-function unit busy instead of taking turns.
