What this record changed: replaced the hand-written one-thread-per-output matmul kernels with cuBLAS /
cuBLASLt tensor-core GEMMs for all projections (QKV, attention output, MLP up/down, logits), routing the
dominant FLOPs onto the tensor cores (TF32 path while storage is still FP32).

Measured numbers (the repo's own development figures; lower time / higher throughput is better):

| comparison | figure | source |
|---|---|---|
| naive hand-written matmul kernel vs. cuBLAS | "version 2 calls cuBLAS, **very fast**"; "version 3 calls cuBLASLt, **should be even faster**" | `dev/cuda/matmul_forward.cu` top-of-file kernel ladder (versions 1→2→3) |
| naive attention path vs. cuBLAS+softmax | the cuBLAS-based attention is **~20× faster** than the naive port | `dev/cuda/attention_forward.cu` (version 3 note: "~20X faster than (1)") |

The matmul-forward development file orders the kernels naive (1) → cuBLAS (2) → cuBLASLt (3) in increasing
speed; the repo does not publish a separate end-to-end tokens/sec number isolated to *only* this swap (the
mainline trainer ships with the full stack), so the per-rung evidence is the kernel-level ordering above. The
20× attention figure is the closest direct naive-vs-library throughput ratio recorded in the repo and bounds
the scale of moving the dominant work off the naive FP32 path onto the library tensor-core kernels.
