Data parallelism scaled the throughput by roughly $N\times$, but it left an $N$-way redundancy I named at the end of the last rung and want to cut. Walk through what each of the $N$ GPUs does at the optimizer step: each holds a full FP32 master copy of every weight, a full FP32 $m$, a full FP32 $v$ — about 12 bytes per parameter of optimizer state, replicated $N$ times. After the all-reduce, each GPU holds the *full* averaged gradient, and each then runs the *identical* AdamW update over *all* parameters, computing the same new value for every weight as every other GPU. $N$ copies of the state, $N$ copies of the gradient, $N$ GPUs doing the same arithmetic $N$ times. For the 124M model the memory is affordable, but the redundancy is real, and it is the kind of waste the rest of this work has been systematically deleting. The optimizer state and the optimizer *work* are the last redundant things in the step.

The observation that breaks the redundancy: after a data-parallel all-reduce, every GPU has the *same* averaged gradient and computes the *same* update, so the work and the state are not just replicated — they are perfectly redundant. I propose **ZeRO-1 optimizer-state sharding**. Partition the parameters into $N$ shards and make GPU $r$ responsible *only* for shard $r$: only GPU $r$ holds the master weights, $m$, and $v$ for shard $r$, and only GPU $r$ runs AdamW on shard $r$. Each GPU then does $1/N$ of the optimizer work and holds $1/N$ of the optimizer state. After each GPU updates its own shard, the GPUs exchange their freshly-updated shards so everyone again has the full set of weights for the next forward. The model stays replicated; the two moments and the master weights — the bulk of the 12-bytes-per-parameter footprint — drop to $1/N$ per GPU.

There is a piece of free lunch hiding in the communication. In plain data parallelism the gradient communication is an *all-reduce*: sum across GPUs and give *every* GPU the full result. But under sharding, GPU $r$ only needs the *averaged gradient for its own shard $r$* — it has no use for the averaged gradient of the other shards, since it will not update them. An all-reduce that hands every GPU the full averaged gradient does more communication than I need. The collective that gives each GPU only *its* slice of the sum is a *reduce-scatter*: it sums the gradient across GPUs (like all-reduce) but *scatters* the result so GPU $r$ receives only shard $r$. And a reduce-scatter moves *less* data than an all-reduce — an all-reduce is essentially a reduce-scatter followed by an all-gather, so by needing only the reduce-scatter half on the gradient I have cut the gradient communication too. Sharding does not cost extra communication on the gradient side; it *reduces* it.

So the full step under ZeRO-1 is: backward produces the full gradient on each GPU; instead of all-reducing it, **reduce-scatter** it so GPU $r$ ends up with the averaged gradient for shard $r$ only; GPU $r$ runs AdamW over shard $r$ (using its $1/N$ of the master weights and moments) to produce the updated weights for shard $r$; then **all-gather** the updated shards so every GPU reassembles the full weight tensor for the next forward. The all-gather is the one piece sharding *adds* relative to plain data parallelism (which did not need to gather weights because every GPU updated all of them), but it moves the BF16 *weights* (2 bytes/param) and replaces the all-reduce's second half — net, the communication is comparable while the state and compute drop to $1/N$.

The reduce-scatter slots straight into the overlapped-background-stream machinery from the data-parallel rung — it is a different NCCL collective behind the same event-gated NCCL stream. The same `multi_gpu_async_reduce_gradient` branches on the ZeRO stage: stage 0 does `ncclAllReduce`, stage 1 does `ncclReduceScatter` into the GPU's shard slice of the gradient buffer, sized `pointers_sizes[i] / num_processes` at offset $\text{shard\_size}\cdot\text{process\_rank}$. So the layer-by-layer overlap of gradient communication with backward compute carries over unchanged — the reduce-scatter for each layer fires in the background stream as soon as that layer's backward finishes, just as the all-reduce did.

The optimizer step computes only this GPU's shard. The update loop walks each parameter tensor, asks for *this rank's shard offset and size* within the tensor, and runs AdamW on just that slice. The master weights and moments are indexed by a *partial* offset (`tensor.offset / num_processes`) because each GPU only allocated $1/N$ of them, while the live BF16 weights and gradients are indexed by the *full* offset plus the shard offset — the GPU does hold the full BF16 weight tensor (it needs it for the forward) but only updates its shard. Right after updating its shard, each GPU all-gathers the updated BF16 weights so every GPU has the full weight tensor before the next forward.

There is a memory dividend beyond the optimizer state itself, and it is what makes the *whole stack* compose into the record. The optimizer footprint dropping from ~12 bytes/param replicated to ~$12/N$ bytes/param frees a large chunk of HBM, and that freed memory can go straight into a *bigger per-GPU micro-batch*. A bigger micro-batch means fewer gradient-accumulation inner loops to reach the same 0.5M-token global batch, which means fewer kernel launches and higher utilization per step — and the matmuls run at larger tile sizes where the tensor cores hit higher efficiency. If memory is still tight, the **recompute** knob from the GELU-fusion rung is available: throw away the GELU (and optionally LayerNorm) forward activations and recompute them during backward, trading a little compute for a lot of activation memory, freeing yet more room for batch size. Sharding, recompute, and the fused kernels all push the same way — fit a bigger batch, run fewer-but-fatter steps, keep the tensor cores saturated — which lifts the MFU into the range where the run is genuinely cheap.

Correctness is again exact. Sharding the optimizer state changes *where* each parameter's update is computed and stored, not *what* it is: GPU $r$ computes the same AdamW update for shard $r$ that a single GPU would, the reduce-scatter produces the same averaged gradient on that shard as the all-reduce would have, and the all-gather reassembles the identical full weight tensor. The optimizer trajectory and the final validation loss are unchanged from plain data parallelism, which was unchanged from single-GPU.

So the stack is complete, top to bottom, every redundancy and every avoidable byte of HBM traffic in a step systematically removed: cuBLAS/cuBLASLt tensor-core GEMMs with fused bias and GELU epilogues both directions; BF16 storage with FP32 master weights and stochastic rounding; 128-bit packed memory I/O on every elementwise kernel; residual+LayerNorm fused; flash attention that never materializes the score matrix; a fused classifier that never materializes the $(B,T,V)$ probabilities; NCCL data-parallel gradient averaging overlapped with backward compute; and ZeRO-1 sharding that cuts the optimizer state and compute to $1/N$, drops the gradient collective to a reduce-scatter, and frees the memory that lets a bigger batch keep the tensor cores saturated. The target this configuration must clear is the one fixed at the start — GPT-2 (124M) reproduced to validation loss $\approx 3.29$ over ~10B tokens (~18,865 steps, ~0.5M-token batch, context 1024) — and it must clear it as fast and as cheaply as the hardware allows: on a node of 8 A100 80GB SXM GPUs, that means driving the run to roughly 60% MFU and finishing in about 90 minutes for about $20, with the same recipe scaling onto 8× H100 to push the throughput further. That is the finish line — not a new model and not a lower loss, but the *same* 3.29, reached at the maximum tokens/sec and MFU this stack can extract from the GPUs. ZeRO-1: the reduce-scatter branch of the gradient collective, and the sharded AdamW step with its all-gather:

```c
// gradient collective: reduce-scatter under ZeRO-1 (each GPU gets only its shard of the averaged grad)
} else if(config->zero_stage == 1) {
    assert(pointers_sizes[i] % config->num_processes == 0);
    size_t shard_size = pointers_sizes[i] / config->num_processes;
    ptrdiff_t shard_offset = (ptrdiff_t)shard_size * config->process_rank;
    ncclCheck(ncclReduceScatter(
            pointers[i], pointers[i] + shard_offset, shard_size,
            ncclFloatX, ncclAvg, config->nccl_comm, config->nccl_stream));
}
```

```c
// sharding helper: rank r owns [r * elements/nproc, size elements/nproc)
ShardInfo multi_gpu_get_shard_offset(size_t elements, const MultiGpuConfig* config, int shard_at_stage) {
    const int nproc = config->num_processes;
    if(config->zero_stage >= shard_at_stage) {
        return {(ptrdiff_t)(config->process_rank * (elements / nproc)), elements / nproc};
    } else {
        return {0, elements};
    }
}
```

```c
// optimizer step: each GPU runs AdamW only on its shard, then all-gathers the updated weights
for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    ShardInfo tensor = gpt2_get_tensor_at_layer(model, 0, i);
    ShardInfo shard   = multi_gpu_get_shard_offset(tensor.size, multi_gpu_config, 1);
    ptrdiff_t local_offset_full    = tensor.offset + shard.offset;                  // full BF16 weights/grads
    ptrdiff_t local_offset_partial = tensor.offset / multi_gpu_config->num_processes; // 1/N optimizer state

    floatX* param_ptr = (floatX*)model->params_memory + local_offset_full;
    floatX* grad_ptr  = (floatX*)model->grads_memory  + local_offset_full;
    ptrdiff_t opt_state_offset = multi_gpu_config->zero_stage < 1 ? local_offset_full : local_offset_partial;
    float* m_ptr = model->m_memory + opt_state_offset;
    float* v_ptr = model->v_memory + opt_state_offset;
    float* master_ptr = (model->master_weights != nullptr) ? model->master_weights + opt_state_offset : nullptr;

    // update only this GPU's shard.size parameters
    adamw_update(param_ptr, master_ptr, grad_ptr, m_ptr, v_ptr,
                 shard.size, tensor.size, tensor.size, shard.size, num_layers,
                 learning_rate, beta1, beta2, t, eps, wd, grad_scale, seed, main_stream);

    if (multi_gpu_config->zero_stage == 1) {
        ncclCheck(ncclGroupStart());
        for(int l = 0; l < num_layers; ++l) {
            // gather updated shards from each process to reassemble the full weight tensor
            ncclCheck(ncclAllGather(param_ptr + l * tensor.size,
                                    (floatX*) model->params_memory + tensor.offset + l * tensor.size,
                                    shard.size, ncclFloatX,
                                    multi_gpu_config->nccl_comm, multi_gpu_config->nccl_stream));
        }
        ncclCheck(ncclGroupEnd());
    }
}
```
