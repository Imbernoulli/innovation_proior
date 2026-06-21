With the classifier fused, every kernel in a single forward+backward is near its peak — matmuls on cuBLAS with fused epilogues, flash attention, fused classifier, vectorized fused elementwise layers. There is no more single-GPU kernel slack worth chasing, yet the run is still long: 10B tokens at a ~0.5M-token batch is about 18,865 optimizer steps, and on one GPU even at high MFU that is many hours. The model is small enough to fit on one device with room to spare, so the limit is not memory — it is that one device can only push so many tokens per second. The obvious untouched lever is parallelism across devices.

Since the model fits on each GPU, the natural parallelism is *data* parallelism: replicate the full model on each of the $N$ GPUs, give each a different slice of the global batch, have each run its own forward+backward to produce its own gradient, then *average the gradients across all GPUs* so every replica takes the same optimizer step and the replicas stay identical. With $N$ GPUs each processing a micro-batch, the effective batch is $N\times$ larger per wall-clock step, so I reach the same 18,865-step / 10B-token budget in roughly $1/N$ the time — *provided the gradient averaging is cheap*. That clause is the whole engineering problem: if the cross-GPU communication is slow or serializes against the compute, the speedup evaporates.

So I propose **NCCL data-parallel training with the all-reduce overlapped with backprop**. The gradient average is an *all-reduce*: every GPU contributes its gradient tensor, the values are summed element-wise across devices (then divided by $N$ for the mean), and the result is broadcast back so every GPU ends with the same averaged gradient. NCCL provides this as a tuned collective (`ncclAllReduce` with `ncclAvg`) that uses the fast inter-GPU interconnect (NVLink within a node) and a ring/tree algorithm — I do not hand-roll cross-GPU communication any more than I hand-rolled the GEMM. To launch $N$ processes, one per GPU, I bootstrap with MPI (`mpirun -np N`), exchange the NCCL communicator id, and initialize NCCL. The skeleton is: each GPU runs forward+backward, then `ncclAllReduce` over the gradient buffers with `ncclAvg`, then the AdamW step on the averaged gradient.

The naive version of this — finish the *entire* backward, then do one big all-reduce over all gradients, then step — works but leaves a large amount on the floor. While the all-reduce runs, the compute units are idle (backward is done, the optimizer cannot start until gradients are averaged); while the backward runs, the interconnect is idle. The two are serialized, so the step time is $\text{backward} + \text{allreduce}$ when it could be closer to $\max(\text{backward}, \text{allreduce})$. For a model whose gradient is hundreds of megabytes, that all-reduce is not negligible.

The fix comes from the structure of backpropagation: gradients become available *layer by layer*, last layer to first, as the backward proceeds. The last layer's gradients are ready while the first eleven layers are still computing. So I do not wait for the whole backward to finish before reducing — I fire off the all-reduce for each layer's gradients *as soon as that layer's backward completes*, and let it run on the interconnect *in the background while earlier layers are still computing*. By the time backward reaches layer 0, most later layers' gradients have already been reduced and are waiting. The all-reduce is hidden behind the backward compute instead of stacked on top of it — the difference between communication being free and communication being a tax.

Making the overlap actually happen requires putting the NCCL collectives on a *separate CUDA stream* from the compute so the scheduler can run them concurrently. But the two streams share data: a layer's all-reduce must not start until that layer's gradient kernels have finished writing. The clean cross-stream dependency is a CUDA *event*: after enqueuing a layer's backward on the compute stream, record an event on it; have the NCCL stream `cudaStreamWaitEvent` on that event before its all-reduce. Using an event rather than a host sync means I never block the CPU — the host keeps enqueuing the next layer's backward kernels onto the compute stream while the NCCL stream consumes the just-finished gradients. The reduce itself I batch with `ncclGroupStart()`/`ncclGroupEnd()`: a layer has a dozen gradient tensors (the two LayerNorms' weight/bias, the QKV weight/bias, the attention-output weight/bias, the two MLP weight/bias), and grouping lets NCCL fuse them into a single collective kernel rather than launching twelve tiny ones, which matters because small collectives are launch-overhead-bound. So per layer, at the end of its backward, the grouped reduce of that layer's gradient bundle runs on the NCCL stream behind the event, while the loop moves on to the previous layer.

A subtlety on *when* this fires: with gradient accumulation (several micro-batches summed before a step to hit the 0.5M-token batch on a smaller per-GPU micro-batch), I only reduce across GPUs on the *last* micro-batch of the accumulation, after the local gradient is fully accumulated — reducing on every micro-batch would multiply the communication for nothing. So the per-layer reduce is gated on `last_step`. Within that final micro-batch, the layer-by-layer overlap stands: each layer's reduce runs behind the earlier layers' backward.

Does this change the result? Data parallelism with gradient averaging is mathematically identical to running the same global batch on one device — the averaged gradient over the $N$ slices equals the gradient over the concatenated batch, so the optimizer trajectory is the same up to floating-point, and the replicas stay bit-identical because they all apply the same averaged-gradient AdamW step. The 3.29 target is unaffected; what changes is wall-clock, which drops by roughly $N\times$ minus whatever overlap fails to hide. On 8 GPUs the ~18,865-step run that was many hours on one device becomes plausibly under two hours, and the documented record is ~3.29 val loss in ~90 minutes on 8× A100 80GB for ~$20.

What this rung does *not* fix is a redundancy I can now see clearly. Every one of the $N$ GPUs holds the *full* optimizer state — FP32 master weights plus AdamW's $m$ and $v$, about 12 bytes per parameter — and after the all-reduce every GPU holds the *full* averaged gradient and runs the *identical* full AdamW step over all parameters, computing the same update for the same parameter as every other GPU. That is $N$-way redundant: $N$ copies of the master weights, $N$ copies of the moments, $N$ GPUs doing the same arithmetic. For the 124M model that is affordable but pure waste, and it is the obvious next thing to cut — the optimizer state and the optimizer work could be *sharded* across the data-parallel group so each GPU owns only its $1/N$ slice. The event-gated, separate-stream, grouped async gradient reduce and its per-layer call site in the backward loop:

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
