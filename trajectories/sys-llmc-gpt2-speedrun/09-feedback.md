What this record changed: added NCCL data-parallel multi-GPU training — replicate the model across N GPUs
(launched via MPI), average gradients with `ncclAllReduce`/`ncclAvg`, and **overlap** the reduce with backprop
by firing each layer's grouped all-reduce on a separate NCCL stream the moment that layer's backward finishes,
gated by a CUDA event (no host sync). Reduce only on the last gradient-accumulation micro-batch.

Measured / documented numbers (the repo's own statements):

| record | figure | source |
|---|---|---|
| multi-GPU launch | `make train_gpt2cu` then `mpirun -np <#GPUs> ./train_gpt2cu` | `README.md` (multi-GPU section) |
| overlap mechanism | "Accumulate gradients from this layer in a **background stream**"; event-gated NCCL stream, no host sync | `train_gpt2.cu`, `llmc/zero.cuh` (`multi_gpu_async_reduce_gradient`) |
| collective | `ncclAllReduce(..., ncclAvg, ...)` for zero_stage 0 | `llmc/zero.cuh` |
| reference run scale | ~10B tokens, ~18,865 steps, ~0.5M-token batch, 8× A100 80GB | GPT-2 (124M) reproduction (Discussion #481) |
| end-to-end result (full 8-GPU stack) | val loss ~3.29 in ~90 minutes on 8× A100 80GB, ~$20 | Discussion #481 |

The repo ships data-parallel multi-GPU as the standard way to run the GPT-2 (124M) reproduction; the documented
end-to-end record (~3.29 val loss in ~90 min on 8× A100 80GB, ~$20) is achieved with the full multi-GPU stack.
The repo does not isolate a 1-GPU-vs-N-GPU tokens/sec table for this specific rung; the per-rung evidence is the
overlapped-all-reduce mechanism itself and the structural near-linear scaling argument (effective batch N× per
step at ~1/N wall-clock when the reduce is hidden behind backward compute). Data-parallel gradient averaging is
mathematically identical to the single-GPU global batch, so the 3.29 target is held exactly (replicas stay
bit-identical).
