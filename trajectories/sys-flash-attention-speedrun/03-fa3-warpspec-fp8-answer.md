**Problem (from step 2).** The well-partitioned FlashAttention-2 kernel hits 72% MFU on A100 but only ~35% of the
H100's tensor-core peak — two-thirds of Hopper idle. The kernel is *synchronous*: one warp loads, then computes
`S`, then does the softmax, then computes `P·V`, each waiting on the last, so Hopper's dedicated async engines —
the TMA copy engine and the asynchronous warpgroup matmul (WGMMA) — are never overlapped with each other or with
the softmax. Everything takes turns. And the FP8 tensor-core path (≈2× the FP16 rate) is unused because naive FP8
quantization of attention's outlier activations destroys accuracy.

**Key idea — FlashAttention-3 (Hopper): warp-specialized producer/consumer pipeline + async TMA + WGMMA, softmax
overlapped with matmul, plus accurate FP8.** (1) **Warp specialization**: dedicate one warpgroup as a *producer*
that only drives TMA bulk-loads of the next K/V tiles into a shared-memory pipeline, and the others as *consumers*
that run WGMMA + softmax — so loading is hidden completely behind compute; the producer deallocates its registers
and donates them to the register-hungry consumers. (2) **Overlap softmax with matmul**: because WGMMA is async,
issue the next block's `QKᵀ` GEMM and let it run on the tensor cores while the current block's softmax runs on the
SFU, pipelining the two matmuls and the softmax across iterations so a matmul is always in flight. (3) **FP8 with
incoherent processing**: run the matmuls in FP8 (e4m3) with per-tensor descales, and first multiply Q and K by a
random orthogonal (Hadamard) matrix `R` — since `(QR)(KR)ᵀ = QKᵀ` the scores are unchanged, but the rotation
spreads each outlier across all coordinates, shrinking the per-vector dynamic range so FP8 quantizes it with far
less error.

**Why it works.** The producer/consumer split, the matmul/softmax pipelining, and the register donation are all
*scheduling* — they reorder independent work and reassign warpgroup roles without changing what is computed, so
FP16/BF16 output stays bit-for-bit exact and memory stays linear; utilization rises from ~35% toward ~75% because
the TMA engine, tensor cores, and softmax units are now busy simultaneously instead of in turns. FP8 preserves
the math too: the Hadamard rotation is exactly invertible inside the score product (`R Rᵀ = I`), the descales are
algebraic rescalings, and the result accumulates to BF16 — so the ≈2× FP8 throughput comes with the quantization
error of the matmuls reduced by incoherent processing rather than the accuracy collapse of naive FP8.

**Realized record (provenance: the repo ships no machine-readable result file for this rung, but the *code* — warp
specialization, TMA, WGMMA, the softmax/matmul overlap, the FP8 descale path — is the in-repo Hopper kernel
excerpted below).** FlashAttention-3 is **1.5–2.0× faster than FlashAttention-2 in FP16, up to 740 TFLOPs/s, i.e.
75% utilization of the H100's theoretical max**; in **FP8 it reaches close to 1.2 PFLOPs/s, with 2.6× smaller
error than baseline FP8 attention** (the incoherent-processing gain). This is the endpoint of the ladder: the
exact attention output of FlashAttention v1, run at three-quarters of a modern datacenter GPU's FP16 peak and
near a petaflop in FP8.

**Change / code.** Warp-specialization dispatch — warpgroup 0 is the TMA producer (registers deallocated), the
rest are consumers (`hopper/flash_fwd_kernel_sm90.h`):

```cuda
        if (warp_group_idx == 0) {  // Producer
            cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
            PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipelineK>();
            // ... loop over work tiles, issuing TMA loads of Q, K, V ...
            mainloop.load(params.mainloop, pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write,
                          shared_storage, scheduler_prefetch, seqlen_info, block_coord, work_idx);
        } else {  // Consumer
            cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
            TiledMmaPV tiled_mma_pv;
            tile_valid = mainloop.mma(
                params.mainloop, pipeline_k, pipeline_v, smem_pipe_read,
                tOrO, softmax, threadIdx.x - MmaThreadOffset, work_idx, seqlen_info, block_coord, shared_storage);
        }
```

The producer's async TMA load — acquire a pipeline stage, fire the copy with its barrier
(`hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`):

```cuda
        auto load_K = [&] (int const n_block, auto const& smem_pipe_write, auto need_seqlenk_masking_type) {
            pipeline_k.producer_acquire(smem_pipe_write);
            copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                tKgK_TMA(_, n_block_idx, bidb_kv_idx), tKsK_TMA(_, smem_pipe_write.index()));
        };
```

The consumer's pipelined step — issue both WGMMAs async, run the softmax overlapping them, FP8 register permute
on the incoherent path (`hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`):

```cuda
            // Each step does gemm0 for iter n_block, gemm1 for iter n_block + 1, and softmax for iter n_block.
            auto fwd_step = [&](int const n_block, auto mask_fn, auto check_inf_type) {
                ++smem_pipe_read;
                if (!UseSchedulerBarrier || warp_group_idx == 0) { consumer_wait(pipeline_k, smem_pipe_read); }
                warp_scheduler_barrier_sync();
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_pv, cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP), tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
                warp_scheduler_barrier_arrive();
                warpgroup_wait<1>();
                pipeline_k.consumer_release(smem_pipe_read);
                mask_fn(tSrS, n_block);
                cute::copy(softmax.template max_get_scale</*Is_first=*/false, Check_inf>(tSrS), scores_scale);
                softmax.template online_softmax</*Is_first=*/false, Check_inf>(tSrS);
                if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
                convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);
                softmax.rescale_o(tOrO, scores_scale);
            };
```

The FP8 path — per-tensor descale at the matmul boundaries, finalize with `v_descale`, accumulate to BF16
(`hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`, `hopper/flash_attn_interface.py`):

```cuda
            float const v_descale = !Is_FP8 || params.ptr_v_descale == nullptr ? 1.0f : params.ptr_v_descale[bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale)];
            cute::copy(softmax.finalize(v_descale), scores_scale);
```

```python
    q_type = q.dtype
    if q_type == torch.float8_e4m3fn:
        out_dtype = torch.bfloat16
    else:
        out_dtype = q_type
```
