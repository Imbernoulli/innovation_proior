The fused kernel gave a real win — the score matrix never touches HBM, memory is linear in sequence length, the backward recomputes instead of storing $P$, and end to end it is about $3\times$ faster on GPT-2 at sequence length 1K, with a 15% wall-clock improvement over the MLPerf-1.1 BERT-large training record, all bit-for-bit exact. But "compute-bound" is not the same as "at peak": on an A100 the kernel reaches only a fraction of the 312 TFLOPs/s the tensor cores can deliver. The matmul math already runs on the tensor cores, so the inefficiency cannot be in the arithmetic — it has to be in *how the work is laid out across the GPU*. Looking at the kernel as a scheduling problem turns up three separate leaks, all about partitioning.

The first leak is the grid. The fused design parallelizes over query blocks and over $(\text{batch}\times\text{heads})$ — one block per $(\text{query-tile},\,\text{batch},\,\text{head})$. That fills the GPU when $\text{batch}\times\text{heads}$ is large, but the regime I care about for long context is small batch and long sequence: $\text{batch}=1$, a few heads, $N=16\mathrm{K}$. Then $\text{batch}\times\text{heads}$ is tiny, the block count is small relative to an A100's ~108 SMs, and most of the chip is idle even though the sequence dimension is enormous. The second leak is the non-matmul FLOPs inside the inner loop: between the two GEMMs, the online softmax does work on the CUDA cores and SFUs that does not overlap the tensor cores and therefore stalls them. In particular the per-iteration rescale-and-normalize of the *whole* output accumulator $\text{acc\_o}$ (a $B_r\times d$ tile) costs $O(B_r\cdot d)$ scalar multiplies on every key block. The third leak only surfaces when I look at the *warps* inside a block: the fused kernel partitions the *key* dimension across warps, so each warp owns a key-slice — but the softmax is a reduction *over keys*, which forces warps holding different key-slices of the same row to exchange partial maxima and sums through shared memory, with a `__syncthreads`, on every key block, and to sum their partial $P\,V$ outputs the same way.

I propose **FlashAttention-2**: the identical exact-attention math, re-partitioned along three axes. First, **parallelize over the sequence dimension** by making the query blocks a first-class grid axis. Launch $\text{num\_m\_block}=\lceil \text{seqlen}_q / B_r\rceil$ blocks on one grid dimension and batch and heads on the others, so the grid is `dim3(num_m_block, batch, heads)` and the total block count scales with sequence length. Different query blocks own disjoint output rows, each streams over all keys and runs its own online softmax, so they never communicate — the parallelization is free, and a single long sequence on one head now saturates the SMs. This is the one-line change with the largest reach.

Second, **drive the non-matmul FLOPs toward the floor**. The division by $\ell$ is the easy half: dividing $\text{acc\_o}$ by the row sum is *linear* and commutes with the accumulation, so there is no reason to apply $1/\ell$ on every iteration. I carry the *unnormalized* output through the whole loop and divide by the final $\ell$ exactly once, in the epilogue — the per-iteration $1/\ell$ work vanishes and only one division per row survives. The max-rescale by $\alpha=e^{m_{\text{old}}-m_{\text{new}}}$ is genuinely needed inside the loop because $m$ changes block to block, but I make it the *only* non-matmul touch of $\text{acc\_o}$ that remains, so each iteration is one GEMM for $S$, one max-reduction, one $\exp$, one rescale of $\text{acc\_o}$, one GEMM for $P\,V$. A useful micro-trick falls out of the $\exp$: computing $\exp_2(x\cdot\log_2 e)$ instead of $\exp(x)$ uses the hardware's fast base-2 exponential on the SFU and lets the compiler fold the scale into a fused multiply-add, so I precompute $\text{softmax\_scale}\cdot\log_2 e$ once and exponentiate in base 2. Every non-matmul cycle removed is a cycle the tensor cores are not stalled.

Third, and the subtlest, **split the work across warps along queries, not keys**. Each warp owns a disjoint slab of the $B_r$ query rows and computes the *full* key range for its own rows. Then a single warp holds an entire query row's scores across all keys in the block, so that row's softmax — max, $\exp$, sum, the online-softmax update, the $P\,V$ accumulation — is *entirely local to the warp*. No warp needs another's partials, so there is nothing to reduce across warps and no barrier in the inner loop; $K$ and $V$ are loaded once into shared memory and every warp reads the same tiles against its own queries, so the shared-memory traffic is not even duplicated. The insight is to divide attention along the dimension the softmax does *not* reduce over (queries) rather than the one it does (keys), which makes each warp's softmax self-contained and deletes the cross-warp reductions and `__syncthreads` that the split-K scheme paid every iteration.

None of this changes the result. The grid covers the same outputs; deferring $1/\ell$ to the epilogue is algebraic because division commutes with the linear accumulation; query-split merely reassigns which warp computes which output row. The output stays bit-for-bit identical and memory stays linear. What changes is utilization — the SMs are saturated even at batch 1, the tensor cores stall less, and the warps stop synchronizing in the hot loop — and together the three changes roughly double the throughput of the same fused kernel, reaching up to ~225 TFLOPs/s = 72% MFU on an A100.

The grid that makes the sequence a parallel dimension (`csrc/flash_attn/src/flash_fwd_launch_template.h`):

```cpp
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    // ... template dispatch over the masking / even-shape / softcap switches ...
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
```

The inner loop — GEMM for `S`, the fused online-softmax rescale of the *output accumulator* (the only non-matmul touch of `acc_o` left inside the loop), GEMM for `P·V` — with the `1/ℓ` normalization deferred to a single post-loop call (`csrc/flash_attn/src/flash_fwd_kernel.h`):

```cpp
        FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );
        // online softmax: rescale running max/sum + output accumulator; NO 1/ℓ division here
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);
        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
        Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
        FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue: the single 1/ℓ normalization for the whole row, after the loop
    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);
```

The mask indexing `m_block * kBlockM + (tidx / 32) * 16 + …` advances the query-row coordinate per warp (`tidx / 32` is the warp index), i.e. the warps are split over the `BLOCK_M` query rows — the key range `n_block` is iterated whole by every warp — so each warp's softmax reduction stays inside the warp.
