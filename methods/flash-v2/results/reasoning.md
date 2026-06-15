Let me start from what actually hurts. I have a fused exact-attention kernel that already solved the memory problem — it tiles Q, K, V into blocks, streams them through SRAM, uses the online-softmax statistics to combine blocks, and never writes the N×N score matrix to HBM, so its memory is linear in N and it does many-fold fewer HBM accesses than the materialized implementation, `Θ(N²d²/M)` against `Θ(Nd + N²)`. That bought 2-4x over standard attention. And yet when I profile it on an A100 the forward pass sits at only about 25-40% of the device's theoretical FLOPs/s, while an optimized GEMM on the same card runs at 80-90% of peak. So the kernel is no longer waiting on memory; the Tensor Cores are idle for some other reason. The whole game now is: keep the exact output and the linear memory, but find where the *work* is being scheduled badly and fix it so I get GEMM-like utilization.

Before I touch anything I need to internalize the cost model, because it tells me what "scheduled badly" even means here. The A100 does 312 TFLOPs/s of FP16/BF16 matmul on its Tensor Cores but only 19.5 TFLOPs/s of non-matmul FP32 on the regular ALUs. Divide those: a non-matmul FLOP is about 16x more expensive than a matmul FLOP. That number reframes the whole problem. In attention the matmuls are QKᵀ and PV; everything else — the exponentials, the row-maxes, the rescalings, the divisions, the mask comparisons — is non-matmul. By raw FLOP count the non-matmul part is small, a couple of N×N elementwise passes against two N×N×d GEMMs. But by *time* it can dominate, because each of those cheap-looking FLOPs costs 16 matmul-FLOPs of wall clock. So my first instinct — "the kernel is exact and IO-optimal, what's left to do?" — is wrong. There's a whole axis I wasn't counting: every elementwise operation in the inner loop is disproportionately expensive, and if I can move non-matmul work out of the hot loop without changing the answer, I directly buy back throughput. Let me hold that thought and go look at the inner loop I actually have.

The forward loop I inherited keeps a running output block O_i together with the online-softmax statistics m_i (running max) and ℓ_i (running normalizer), and on every K/V block j it does the combine-and-rescale. Written out, the update is `O_i ← diag(ℓ_i^new)^{-1}(diag(ℓ_i) e^{m_i - m_i^new} O_i + e^{m̃_{ij} - m_i^new} P̃_{ij} V_j)`. Stare at that. Inside the loop, every single block, I divide the running output by the new normalizer `diag(ℓ_i^new)^{-1}`. That's a per-element division over the whole B_r×d output tile, on the regular ALUs, once per column block — and a division is exactly the 16x-expensive kind of op I just decided to hunt. Why am I dividing every block? Only because I insist the running O_i be the *correctly normalized* partial output at every step. But I don't need it to be. I only need the final O_i, at the very end of the loop, to be normalized. This is the deferred-normalization idea — accumulate the output *unnormalized* and divide once. Let me check it actually works with the online max correction, not just hand-wave it.

Take the simplest case, two blocks, and be careful with the max-rescaling because that's where these things break. Block 1: m⁽¹⁾ = rowmax(S⁽¹⁾), ℓ⁽¹⁾ = rowsum(e^{S⁽¹⁾ - m⁽¹⁾}), and instead of forming the normalized P̃⁽¹⁾V₁ I keep the unnormalized accumulator Õ⁽¹⁾ = e^{S⁽¹⁾ - m⁽¹⁾} V₁. Block 2: the new max m⁽²⁾ = max(m⁽¹⁾, rowmax(S⁽²⁾)) = m. The normalizer combines with the online-softmax correction, ℓ⁽²⁾ = e^{m⁽¹⁾ - m⁽²⁾} ℓ⁽¹⁾ + rowsum(e^{S⁽²⁾ - m⁽²⁾}), which equals rowsum(e^{S⁽¹⁾ - m}) + rowsum(e^{S⁽²⁾ - m}) = ℓ, the right full-row normalizer. Now the accumulator: I rescale the old unnormalized accumulator by just the max-correction factor and add the new unnormalized contribution, Õ⁽²⁾ = diag(e^{m⁽¹⁾ - m⁽²⁾}) Õ⁽¹⁾ + e^{S⁽²⁾ - m⁽²⁾} V₂. Substitute Õ⁽¹⁾: that's e^{m⁽¹⁾ - m} e^{S⁽¹⁾ - m⁽¹⁾} V₁ + e^{S⁽²⁾ - m} V₂ = e^{S⁽¹⁾ - m} V₁ + e^{S⁽²⁾ - m} V₂ — the two exponents both telescope down to the global max m, exactly as they should. And only at the end do I normalize: O⁽²⁾ = diag(ℓ⁽²⁾)^{-1} Õ⁽²⁾ = (e^{S⁽¹⁾-m}V₁ + e^{S⁽²⁾-m}V₂)/ℓ = O. So it's exact. What did deferring cost me, and what did it save? The accumulator still has to be rescaled by the max-correction factor `diag(e^{m⁽¹⁾ - m⁽²⁾})` each block — that's unavoidable, because the running max really can jump on any block and the past sum was taken relative to the old max. But the *division* by the normalizer is gone from the loop entirely; it happens once, after the last block. One division per output row per loop, instead of one per output row per *block*. That's `T_c` divisions collapsed to one. Good — that's the first chunk of non-matmul work out of the hot path, for free, with the answer unchanged.

While I'm in the statistics, there's a second little economy. For the backward pass I need to be able to recompute P = softmax(S), and naively I'd save both m and ℓ for that. But P = e^{S - m}/ℓ = e^{S - m - log ℓ} = e^{S - (m + log ℓ)}. So in the mathematical, natural-exponential view I only need the single logsumexp L = m + log(ℓ); one vector replaces two. In the base-2 kernel, the same object is stored in base-2 units as M = m₂ + log2(ℓ), and the backward recomputes P = exp2(S₂ - M). Half the statistics memory, and it's the natural object anyway — the row's log-partition, just expressed in the base the kernel is using. Store that per row block; drop the separate m and ℓ.

Now the exponentials themselves. The inner loop calls exp on a B_r×B_c tile every block — that's a lot of the non-matmul time, and it's transcendental, the most expensive ALU work there is. I can't remove the exponentials, the softmax needs them. But the GPU's special-function units compute base-2 exponent, exp2, natively and faster than natural exp. And e^x = 2^{x·log₂e} with log₂e = 1.44269504, so I can convert: instead of e^{S - m} I compute exp2((S - m)·log₂e). At first that looks like I just *added* a multiply by 1.44269504 to every element — trading the cost of natural exp for exp2 plus a per-element scale. But watch where the scale can go. The scores are S = (QKᵀ)·sm_scale, and I'm about to multiply S by log₂e anyway. Both sm_scale and log₂e are constants for the whole kernel. So fold them together and push them onto Q *at load time*: load q′ = q·(sm_scale·log₂e) once, before the loop even starts. Then QKᵀ already comes out as S·log₂e, the inner loop is pure exp2 with no extra per-element multiply, and the running max comparison and the subtraction all happen in the log₂ scale consistently. The scale multiply, which used to be one multiply per element of the N×N score matrix, becomes one multiply per element of the B_r×d query tile, done once. exp becomes exp2 (cheaper SFU op) and the scaling leaves the hot loop. Another slice of non-matmul work, gone, answer unchanged.

Causal masking is the next non-matmul sink, and it's worse than it looks because it's a comparison plus a conditional write `tl.where(row ≥ col, qk, -inf)` over the whole tile, plus then exponentiating a tile full of -inf to zero. But I'm already working in blocks, so I can reason about masking at the block level instead of the element level. Fix a query row block i (rows in `[start_m·B_r, (start_m+1)·B_r)`). A K/V column block j covers columns `[j·B_c, (j+1)·B_c)`. There are three kinds of blocks. If the block is entirely above the diagonal — every column index exceeds every row index in the tile, i.e. j·B_c ≥ (start_m+1)·B_r — then the whole tile is masked to zero and contributes nothing: I can skip the block outright, no QKᵀ, no exp, no PV. For a causal pattern that's roughly half the blocks for a long sequence, so skipping them is close to a 2x reduction in *both* matmul and non-matmul work on the upper triangle (call it ~1.7-1.8x in practice). If the block is entirely below the diagonal — every column index is ≤ every row index, i.e. (j+1)·B_c ≤ start_m·B_r — then nothing is masked and I can skip the `tl.where` entirely: just compute the block normally. Only the blocks straddling the diagonal actually need the elementwise mask, and for square blocks that's exactly one block per row block. So instead of masking every block, I split the column loop into two passes: a first pass over the strictly-below-diagonal blocks with no mask at all, and a second pass over just the boundary block(s) where I apply `tl.where`. Two passes, mask applied to a vanishing fraction of the work, and the whole upper triangle never computed. The elementwise masking cost essentially disappears.

That's lever one fully worked out — every move was "find a non-matmul operation in the inner loop and either defer it to the end, convert it to a cheaper unit, or skip it by blocks," and every move provably leaves the output exact. But reducing non-matmul FLOPs only helps if the device is busy in the first place, and I have a nagging second observation from the profile: low occupancy. So let me look at how the work is spread across the SMs, because that's a different bottleneck entirely.

The kernel I inherited launches one thread block per attention head, and parallelizes over batch × heads. On an A100 there are 108 SMs, and a thread block runs on one SM, so I want at least ~108 thread blocks (more, to hide latency) to fill the machine. Batch × heads is fine when the batch is big. But the whole reason I care about this kernel is *long* sequences, and long sequences usually come with *small* batches — you can't fit many 16k-token sequences in memory at once. So exactly in the regime I'm optimizing for, batch × heads can drop below 108, and most of the GPU sits idle. I need another dimension to parallelize over. What's available? The sequence length. Here's the structural fact that makes it possible: softmax couples a row across the *key* dimension — to normalize row i I need all of row i's scores — but different *query rows are completely independent*. Row i's output depends on Q_i, all of K, all of V, but not on any other query row. So I can hand each query row block to its own thread block and they never need to talk to each other.

But the loop I have is structured the wrong way for that. Its outer loop runs over K/V *column* blocks j and the inner loop over Q *row* blocks i, which is why it keeps reloading the running O_i and its statistics from HBM — different column-block iterations of the outer loop touch the same row block, so the partial output has to live in HBM between them. If instead I make the outer loop run over Q *row* blocks i, and the inner loop over K/V column blocks j, then each outer iteration is a self-contained computation of one output row block: load Q_i once, loop over all of K and V accumulating the running (Õ_i, m_i, ℓ_i) entirely in SRAM/registers, normalize, write O_i once. No row block's partial output ever round-trips to HBM mid-computation, and — the payoff — each outer iteration is *independent*, so I can give each one its own thread block. The launch grid becomes (number of row blocks) × (batch) × (heads). For a long sequence, the row-block factor is large (16k / 128 = 128 row blocks per head), so even with batch 1 and a handful of heads I now have hundreds of independent thread blocks filling the 108 SMs. The outer loop is embarrassingly parallel; the seqlen dimension restores occupancy precisely when batch × heads alone couldn't. This loop swap — outer over rows, inner over columns, and parallelize across the row blocks — is exactly the reordering that makes both the no-round-trip property and the high occupancy fall out at once.

Now I have the SMs full and the inner loop stripped of cheap-but-slow ops. One layer left: inside a single thread block, how do its warps divide the work? A thread block has 4 or 8 warps, and warps communicate only through shared memory with a sync in between, which is itself overhead I'd like to avoid. The scheme I inherited is "split-K": within the block, K and V are split across the warps and Q is shared by all of them. So each warp computes a partial QKᵀV using its slice of K and V — but its slice covers only some of the columns of the row, so no warp has the *full* row's output; the warps' partial results have to be written out to shared memory, the block synchronizes, and then the partials are summed to get the block's output. Every block, every warp writes its partial to shared memory and reads back others' — exactly the shared-memory round-trips I keep finding are pure overhead. Why is it laid out this way? Because in the old outer-loop-over-K structure, K was the dimension being streamed, so it felt natural to split it. But I just swapped the loops. Now the row is the independent unit, so let me split the *rows* across warps instead — "split-Q": split Q across the warps, keep K and V shared by all of them. Then warp w owns a contiguous slice of the row block's rows. It computes its slice of QKᵀ against the shared K, runs its own online softmax on its own rows (each warp's rows are independent — that's the whole point), and multiplies its P slice by the shared V to get its slice of the output. No warp ever needs another warp's partial result, because rows don't couple. The block's output is just the concatenation of the warps' slices — no shared-memory reduction, no extra sync, no round-trip. The reason split-Q works and split-K didn't is the same reason the loop swap works: the row is the axis of independence, so partitioning along rows means warps never have to combine. Split-K partitioned along the contraction axis, which is exactly the axis that *does* couple, which is why it forced a reduction.

So three levers, all preserving the exact output: fewer non-matmul FLOPs in the inner loop (defer normalization, store only logsumexp, exp2 with the scale folded into Q, two-pass causal that skips half the blocks); parallelize the outer loop over row blocks across the sequence dimension to fill the SMs; and split Q across warps to kill the inter-warp shared-memory reduction. There's one knob left to set, the block sizes B_r×B_c. Bigger blocks mean fewer iterations of the loop and so fewer shared-memory loads/stores per output element — good — but they need more registers and more shared memory to hold the tiles, and past a point the register file spills (catastrophic slowdown) or the shared memory simply doesn't fit and the kernel won't launch. The sweet spot depends on the head dimension d, because d sets how wide each tile is, and on how much shared memory the device has. In practice there are only about four sensible choices, {64,128}×{64,128}, and I pick per head dimension — smaller d can afford a larger B_r (e.g. 128×64), large d (256) needs the smaller 64×64 to stay within registers and shared memory. I'll tune these by hand per head dim for now; there are few enough cases that it's not worth autotuning yet, though autotuning the block sizes is the obvious next step.

Let me make sure I haven't broken correctness anywhere, because every one of these was justified as "answer unchanged." The deferred normalization I proved exact above by telescoping the max corrections. The logsumexp substitution is an algebraic identity, P = e^{S - (m + log ℓ)}, or in the base-2 kernel P = exp2(S₂ - (m₂ + log2 ℓ)). The exp2 conversion is the identity e^x = 2^{x log₂e}, with the scale folded into Q so QKᵀ comes out already in the right base. The two-pass causal only *skips* blocks that are entirely masked (contribute zero) and only *omits the mask* on blocks where no element would be masked, so it computes the same thing as masking every block. The loop swap and the warp split-Q rearrange *where* arithmetic happens, not *what* arithmetic — each output row is still softmax(Q_i Kᵀ) V over all of K, V. So the whole thing returns O = softmax(QKᵀ)V exactly, in O(N²d) FLOPs, with O(N) extra memory for the stored logsumexp — same correctness and memory guarantees as the IO-aware predecessor, just scheduled to keep the Tensor Cores busy.

Now let me write the forward kernel I'd actually ship, in the tiling DSL, filling the empty kernel body and grid/block choice in the harness. One program owns one query row block of one head; it loops over K/V column blocks, keeping the unnormalized accumulator and the online statistics in registers; causal is two passes; exp2 with the scale pre-folded into q.

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
    # One program = one query ROW block of one head. The grid ranges over row
    # blocks (sequence length) and batch*heads, so row blocks are the unit of
    # parallelism -> high occupancy even when batch*heads is small.
    start_m = tl.program_id(0)          # which row block
    off_hz = tl.program_id(1)           # which (batch, head)

    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)   # this program's query rows
    offs_n = tl.arange(0, BLOCK_N)                       # key/value columns within a block
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load Q once; it stays resident for the whole loop. Fold the softmax scale
    # and log2(e) into this Q tile, so the inner loop is pure exp2 with no
    # per-score scale multiply.
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    q = (q * (sm_scale * 1.44269504)).to(tl.float16)

    # Online-softmax running statistics + UNNORMALIZED output accumulator.
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")   # running row max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0            # first alpha=0 makes this a zero seed
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)    # unnormalized O (divide once at end)

    # Causal -> two passes. Pass 1: blocks strictly below the diagonal (no mask
    # needed). Pass 2: the boundary block(s) on the diagonal (mask applied).
    # Blocks entirely above the diagonal are never iterated -> ~half the work skipped.
    if IS_CAUSAL:
        # floor to a BLOCK_N multiple so pass 1 covers only whole below-diagonal blocks
        non_causal_end = (start_m * BLOCK_M // BLOCK_N) * BLOCK_N
    else:
        non_causal_end = seqlen

    # --- Pass 1: no-mask blocks (strictly below the diagonal; all columns valid) ---
    for start_n in range(0, non_causal_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        qk = tl.dot(q, tl.trans(k))                 # already in log2 scale (scale folded into q)
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)           # max-correction factor for the accumulator
        p = tl.math.exp2(qk - m_new[:, None])       # exp2: native, cheap on the SFU
        l_i = l_i * alpha + tl.sum(p, axis=1)       # rescale old normalizer, add new mass
        acc = acc * alpha[:, None]                  # rescale unnormalized accumulator (no divide!)
        v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)             # accumulate unnormalized P@V
        m_i = m_new

    # --- Pass 2: boundary blocks (need the causal mask) ---
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
        qk = tl.where(valid_cols & causal_cols, qk, float("-inf"))  # mask only here
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

    # Normalize ONCE, at the very end, then store output plus base-2 logsumexp.
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
    # Per-headdim block sizes: larger blocks for smaller heads; small head dim
    # 256 must shrink the row block to stay within registers / shared memory.
    if headdim <= 64:
        BLOCK_M, BLOCK_N = 128, 64
    elif headdim <= 128:
        BLOCK_M, BLOCK_N = 128, 64
    else:
        BLOCK_M, BLOCK_N = 64, 64
    # Grid: parallel over ROW blocks (sequence length) AND batch*heads.
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

So the causal chain, end to end. I started with a fused attention kernel that was already memory-IO-optimal but stalled at a quarter of the device's matmul peak while GEMM hits 80-90% — meaning the bottleneck had moved from HBM traffic to how the work is scheduled. The cost model said the scarce resource is non-matmul throughput (each non-matmul FLOP ≈ 16x a matmul FLOP on Tensor Cores), so the first lever was to strip non-matmul work out of the inner loop without changing the answer: accumulate the output unnormalized and divide once at the end (proved exact by telescoping the online-max corrections), store only the logsumexp instead of both the max and the normalizer, compute exp2 instead of exp with the softmax scale and log₂e folded onto Q at load so the hot loop carries no extra multiply, and split the causal loop into a no-mask pass plus a tiny boundary pass while skipping the entirely-masked upper-triangle blocks. The second lever fixed occupancy: because query rows are mutually independent (softmax couples keys within a row, not rows to each other), I swapped the loops so the outer loop runs over query row blocks and parallelized that outer loop across the sequence dimension on top of batch and heads — turning each row block into an independent thread block, which fills the 108 SMs even when long sequences force a small batch, and incidentally stops the running output from round-tripping to HBM. The third lever removed intra-block overhead: split Q across warps instead of splitting K (the rows are the axis of independence, so warps partitioned by rows never have to combine results), eliminating the shared-memory reduction that split-K forced. Block sizes {64,128}×{64,128} are chosen per head dimension to trade fewer loop iterations against register and shared-memory pressure. Every change preserves the exact output and the linear memory, and together they push exact attention toward GEMM-level utilization.
