**Problem (from step 8).** Every single-GPU kernel is near peak, but 10B tokens ≈ 18,865 steps is many hours on
one device. The model fits per-GPU, so the limit is throughput, not memory — and the obvious unused lever is
parallelism across devices.

**Key idea.** **Data parallelism**: replicate the model on N GPUs, give each a slice of the global batch, and
average the per-GPU gradients with an **NCCL all-reduce** (`ncclAllReduce`, `ncclAvg`, over NVLink) so all
replicas take the same step. Effective batch is N× per step → ~1/N wall-clock — *if* the communication is hidden.
Since backprop produces gradients last-layer-first, **overlap** the reduce with compute: fire each layer's
all-reduce on a **separate NCCL stream** the moment that layer's backward finishes, gated by a CUDA **event** so
the host never blocks; batch a layer's dozen gradient tensors with `ncclGroupStart/End`.

**Why it works.** The averaged gradient over N slices equals the gradient over the concatenated batch, so the
optimizer trajectory (and 3.29) is unchanged and replicas stay bit-identical. Overlap turns step time from
`backward + allreduce` toward `max(backward, allreduce)` — the reduce rides on the interconnect while earlier
layers still compute. The event (not a host sync) lets the CPU keep enqueuing; grouping fuses many small
collectives into one. Reduce only on the last gradient-accumulation micro-batch. Still redundant: every GPU
stores the *full* optimizer state (FP32 master + m + v, ~12 B/param) and runs the *identical* full AdamW step —
N-way waste to cut next.

**Change / code.** The event-gated, separate-stream, grouped async gradient reduce, and its per-layer call site
in the backward loop (`llmc/zero.cuh`, `train_gpt2.cu`):

```c
// Block NCCL stream until compute_stream is done, then aggregate pointers in an NCCL group.
template<int N>
void multi_gpu_async_reduce_gradient(
        floatX* const (&pointers)[N], const size_t (&pointers_sizes)[N],
        MultiGpuConfig* config, cudaStream_t compute_stream) {
    if (config->num_processes == 1) { return; } // no multi-GPU, just exit.
    // mark an event on the compute stream and wait on it in the nccl stream: the nccl stream won't start
    // before all compute kernels submitted so far have finished. Using an event (not cudaSyncStream) avoids
    // a host sync, so we can enqueue new work to the GPU right away.
    cudaCheck(cudaEventRecord(config->compute_nccl_sync, compute_stream));
    cudaCheck(cudaStreamWaitEvent(config->nccl_stream, config->compute_nccl_sync));
    ncclCheck(ncclGroupStart()); // aggregate all pointers in a single NCCL GPU kernel
    for (int i = 0; i < N; ++i) {
        if(config->zero_stage == 0) {
            ncclCheck(ncclAllReduce(pointers[i], pointers[i], pointers_sizes[i],
                                    ncclFloatX, ncclAvg, config->nccl_comm, config->nccl_stream));
        }
        // (zero_stage == 1 reduce-scatter handled in the next rung)
    }
    ncclCheck(ncclGroupEnd());
}
```

```c
// in the backward loop, right after this layer's backward kernels (train_gpt2.cu):
layernorm_backward(dresidual, dl_ln1w, dl_ln1b, scratchF, dl_btc, residual,
                   l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C, main_stream);
// Accumulate gradients from this layer in a background stream.
if(last_step) {   // only on the final gradient-accumulation micro-batch
    floatX* const pointers[] = {
        dl_ln1w, dl_ln1b, dl_qkvw, dl_qkvb, dl_attprojw, dl_attprojb,
        dl_ln2w, dl_ln2b, dl_fcw, dl_fcb, dl_fcprojw, dl_fcprojb };
    const size_t nelem[] = {
        C, C, 3*C*C, 3*C, C*C, C, C, C, 4*C*C, 4*C, C*4*C, C };
    multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, main_stream);
}
```
