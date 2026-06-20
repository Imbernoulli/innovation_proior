**Problem (from step 1).** The fused kernel is compute-bound and exact, but it reaches only a fraction of the
A100's tensor-core peak. The slack is in *work partitioning*, not arithmetic: (a) the grid parallelizes only over
query blocks × batch × heads, so at small batch / long context too few blocks launch and the SMs sit idle; (b)
the inner loop spends non-matmul FLOPs — rescaling the whole output accumulator and dividing by the running sum
every key block — that stall the tensor cores; (c) the warps inside a block split the key dimension, so the
softmax (a reduction over keys) forces cross-warp reductions and `__syncthreads` every iteration.

**Key idea — FlashAttention-2: same math, better work partitioning.** Three partitioning changes. (1) **Parallelize
over the sequence dimension**: make the query/sequence blocks a first-class grid axis (`dim3(num_m_block, batch,
heads)`) so a single long sequence on one head still fills the GPU. (2) **Fewer non-matmul FLOPs**: defer the
`1/ℓ` softmax normalization out of the inner loop to a single epilogue division, keep only the unavoidable
max-rescale of the accumulator inside, and use base-2 `exp2` so the SFU/compiler fuse the scale into an FMA. (3)
**Split work across warps along queries, not keys**: each warp owns a slab of query rows and computes the full key
range for them, so its softmax is entirely local — no cross-warp reduction, no barrier in the hot loop.

**Why it works.** All three are scheduling decisions that leave the computed output identical: the grid covers the
same outputs; deferring the `1/ℓ` division is algebraic (division commutes with the linear accumulation); the
query-split just reassigns output rows to warps. So the result stays bit-for-bit exact and linear in memory. What
changes is utilization — the SMs are saturated even at batch 1, the tensor cores stall less because the
non-matmul fraction of the loop is minimized, and the warps stop synchronizing because the softmax reduction now
lives inside a single warp. Together they roughly double throughput over FlashAttention v1.

**Change / code.** The grid that makes the sequence a parallel dimension
(`csrc/flash_attn/src/flash_fwd_launch_template.h`):

```cpp
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    // ... template dispatch over the masking / even-shape / softcap switches ...
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
```

The inner loop — GEMM for `S`, the fused online-softmax rescale of the *output accumulator* (the only non-matmul
touch of `acc_o` left inside the loop), GEMM for `P·V` — with the `1/ℓ` normalization deferred to a single
post-loop call (`csrc/flash_attn/src/flash_fwd_kernel.h`):

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

The mask indexing `m_block * kBlockM + (tidx / 32) * 16 + …` advances the query-row coordinate per warp
(`tidx / 32` is the warp index), i.e. the warps are split over the `BLOCK_M` query rows — the key range `n_block`
is iterated whole by every warp — so each warp's softmax reduction stays inside the warp.
