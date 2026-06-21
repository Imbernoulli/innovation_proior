The re-partitioned kernel reaches 72% MFU on an A100 — near peak for Ampere — but the number that should make me uncomfortable is what happens when I move that same well-tuned FlashAttention-2 kernel to Hopper: it reaches only about 35% of the H100's tensor-core peak. Two-thirds of the new chip sits idle. The kernel did not get worse; the hardware changed underneath it. Hopper added a **Tensor Memory Accelerator (TMA)** that bulk-copies a whole tile between HBM and shared memory as a single asynchronous instruction; **WGMMA** (warpgroup matrix-multiply-async), a tensor-core GEMM issued by a 128-thread warpgroup that reads operands straight from shared memory and runs asynchronously so the issuing threads fire it and move on; and an FP8 (e4m3) tensor-core path at roughly double the FP16 rate. The common thread is *asynchrony* — Hopper wants data movement, matmuls, and the softmax all in flight at once on different engines. The FA2 kernel is synchronous: a warp loads, then computes $S$, then does the softmax, then computes $P\,V$, each waiting on the last. The TMA engine idles while the warps compute, the tensor cores idle while the warps do softmax, and the warps idle while waiting on loads. The 35% is the cost of taking turns.

I propose **FlashAttention-3**, a restructuring of the same exact attention for Hopper's asynchronous machinery, built on three composing ideas. The first is **warp specialization**, a producer/consumer split. Rather than every warpgroup running the full load-then-compute cycle, I dedicate one warpgroup as the *producer*: its only job is to drive the TMA engine, issuing asynchronous loads of the next $K$ and $V$ tiles into shared memory and staying ahead of the rest. The other warpgroups are *consumers*: they do no loading, they wait for a tile to be ready, then run the WGMMA matmuls and the softmax on it. Producer and consumers run at the same time — while the consumers crunch tile $j$ on the tensor cores, the producer is already TMA-loading tile $j{+}1$ — so data movement is hidden completely behind compute, exactly what the synchronous kernel failed to do. A circular buffer of shared-memory stages guarded by a pipeline of barriers connects them: the producer `acquire`s an empty stage, fires the TMA copy with a barrier token, and `commit`s; a consumer `wait`s on that stage, uses the data, and `release`s it. There is a register subtlety that makes the split pay: the producer runs almost no arithmetic and needs few registers, the consumers are register-hungry, and Hopper lets a warpgroup deallocate registers and donate them to others. So the producer calls `warpgroup_reg_dealloc` and the consumers `warpgroup_reg_alloc`, shrinking the producer's footprint to near nothing and handing those registers to the consumers, which lets them keep larger tiles resident and run wider — the specialization puts the register budget where the compute is.

The second idea is **overlapping the softmax with the matmuls inside the consumer**, the subtler win, because even with the loads hidden a consumer still alternates between two kinds of work on different units. The inner loop per key block is WGMMA for $S=QK^\top$ (tensor cores), then softmax — max, $\exp$, sum, the online-softmax rescale (CUDA cores / SFU), then WGMMA for $P\,V$ (tensor cores). Run strictly in order, the tensor cores idle during the softmax and the SFU idles during the GEMMs. But WGMMA is asynchronous, and the next block's $QK^\top$ depends only on $Q$ (resident) and the next $K$ tile (already loaded by the producer), not on the current block's softmax — so I can *issue* the next block's $QK^\top$ and let it run on the tensor cores while the current block's softmax runs on the SFU. Pipelined across iterations, each step issues $\text{gemm}(QK^\top)$ for block $n{+}1$, does the softmax for block $n$ overlapping that matmul, then issues $\text{gemm}(P\,V)$ for block $n$ and the rescale, so some tensor-core work is always in flight while the softmax runs. With two consumer warpgroups this can be taken further into a ping-pong where one warpgroup's softmax overlaps the other's GEMMs, scheduled by named barriers. The point is to never let the tensor cores stall on the softmax.

The third lever does touch the numbers, so it needs care: **FP8**. Hopper's FP8 tensor cores run at roughly double the FP16 rate, so doing the matmuls in FP8 nearly doubles throughput again — but FP8 e4m3 has about three mantissa bits, and attention activations carry occasional large-magnitude *outliers*. Quantizing naively means the outliers either saturate or force a coarse scale that crushes the precision of the ordinary entries, and the error blows up. I handle this with two pieces. The first is per-tensor **descaling**: keep the FP8 values scaled into range and carry scale factors $q_{\text{descale}}, k_{\text{descale}}, v_{\text{descale}}$ applied at the matmul boundaries — the $QK^\top$ product is in units of $q_{\text{descale}}\,k_{\text{descale}}$ and the $P\,V$ product in units of $v_{\text{descale}}$ — so folding these scalars into the softmax scaling and the final normalization recovers the correctly-scaled output, with the FP8 GEMMs fast and the descales a handful of scalar multiplies; because FP8 cannot faithfully represent the result, accumulation and the emitted output are in BF16, not FP8. The second piece, the real idea for the outliers, is **incoherent processing**. The trouble with an outlier is that it is concentrated — one coordinate is huge, the rest small, and FP8 cannot serve both at one scale. So I first multiply $Q$ and $K$ each by a random orthogonal matrix $R$. Because $R$ is orthogonal, $R R^\top = I$ and
$$(QR)(KR)^\top = Q\,R R^\top K^\top = Q K^\top,$$
the scores — and hence the attention output — are *unchanged*. But the random rotation *spreads* each outlier's magnitude across all coordinates of the rotated vector: a single large entry becomes many moderate ones, the per-vector dynamic range shrinks, the values become "incoherent" (no coordinate dominates), and FP8 quantization of the rotated vectors loses far less. I use a fast structured Hadamard transform, which is $O(d\log d)$ and nearly free, as $R$, so the matmul math is exactly preserved while the quantization error of the matmuls is cut by a large factor.

None of this changes the result. The producer just loads the same tiles earlier; the consumers run the same two GEMMs and the same online softmax, only reordered so independent work overlaps; warp specialization is a scheduling of which warpgroup does what; the Hadamard rotation is exactly invertible inside the score product; the descales are algebraic; and FP8 accumulates to BF16. The output is bit-for-bit the same exact $\mathrm{softmax}(QK^\top/\sqrt d)\,V$ as FlashAttention v1 and v2 in FP16/BF16. What changes is that the TMA engine, the tensor cores, and the softmax units are all busy at once instead of taking turns, so utilization climbs from ~35% toward ~75% of the H100's FP16 peak: **FlashAttention-3 is 1.5–2.0× faster than FlashAttention-2 in FP16, up to 740 TFLOPs/s (75% of the H100's theoretical max), and in FP8 reaches close to 1.2 PFLOPs/s with 2.6× smaller error than baseline FP8 attention** — the exact attention output of v1, run at three-quarters of a modern datacenter GPU's FP16 peak and near a petaflop in FP8.

Warp-specialization dispatch — warpgroup 0 is the TMA producer (registers deallocated), the rest are consumers (`hopper/flash_fwd_kernel_sm90.h`):

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

The producer's async TMA load — acquire a pipeline stage, fire the copy with its barrier (`hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`):

```cuda
        auto load_K = [&] (int const n_block, auto const& smem_pipe_write, auto need_seqlenk_masking_type) {
            pipeline_k.producer_acquire(smem_pipe_write);
            copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                tKgK_TMA(_, n_block_idx, bidb_kv_idx), tKsK_TMA(_, smem_pipe_write.index()));
        };
```

The consumer's pipelined step — issue both WGMMAs async, run the softmax overlapping them, FP8 register permute on the incoherent path (`hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`):

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

The FP8 path — per-tensor descale at the matmul boundaries, finalize with `v_descale`, accumulate to BF16 (`hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`, `hopper/flash_attn_interface.py`):

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
