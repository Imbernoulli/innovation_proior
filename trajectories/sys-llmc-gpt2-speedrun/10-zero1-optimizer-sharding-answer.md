**Problem (from step 9).** Plain data parallelism leaves an N-way redundancy: every GPU holds the full FP32
optimizer state (master weights + `m` + `v`, ~12 B/param), holds the full averaged gradient after the
all-reduce, and runs the *identical* full AdamW step over all parameters. State, gradient, and optimizer compute
are all replicated N times — the last redundancy in the step.

**Key idea (finale).** **ZeRO-1**: shard the optimizer state and the optimizer work across the data-parallel
group. GPU `r` owns only shard `r` — only it holds the master weights / `m` / `v` for shard `r` and only it runs
AdamW on shard `r`. The gradient collective drops from **all-reduce** to **reduce-scatter** (each GPU gets only
its shard of the averaged gradient — and reduce-scatter moves *less* data than all-reduce); after each GPU
updates its shard, an **all-gather** reassembles the full BF16 weights for the next forward. Optimizer state and
compute drop to 1/N; the freed HBM funds a **bigger per-GPU micro-batch** (fewer grad-accum loops, fatter
tensor-core tiles, higher MFU), with the GELU/LayerNorm **recompute** knob available if memory is still tight.

**Why it works.** Sharding changes *where* each update is computed and stored, not *what*: GPU `r` computes the
same AdamW update for shard `r` a single GPU would, reduce-scatter yields the same averaged shard gradient as
all-reduce, all-gather reassembles the identical weights. So the optimizer trajectory and final loss are
unchanged — 3.29 holds exactly. The reduce-scatter reuses the event-gated background-stream overlap from step 9.
Sharding + recompute + the fused kernels all push the same way: fit a bigger batch, run fewer-but-fatter steps,
keep the tensor cores saturated — which lifts MFU into the cheap regime.

**Target this configuration must clear.** The fixed task: GPT-2 (124M) reproduced to validation loss ≈ 3.29 over
~10B tokens (~18,865 steps, ~0.5M-token batch, context 1024) — at maximum tokens/sec and MFU. On 8× A100 80GB
SXM that is ~60% MFU, ~90 minutes, ~$20; the same recipe scales onto 8× H100 for higher throughput still. Not a
new model, not a lower loss — the *same* 3.29, as fast and cheap as the hardware allows.

**Change / code.** ZeRO-1: the reduce-scatter branch of the gradient collective, and the sharded AdamW step with
its all-gather (`llmc/zero.cuh`, `train_gpt2.cu`):

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
