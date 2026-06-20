What this record changed: fused the residual-add and the LayerNorm into a single kernel
(`fused_residual_forward`), eliminating the HBM round-trip of the residual sum at every LayerNorm seam (two per
layer × 12 layers). The fast variant uses a warp per token, stages the LayerNorm weight/bias and the residual
row in shared memory, does the mean/variance reductions in FP32, and reads/writes 128-bit packets.

Measured numbers (the repo's own development figures; lower time / higher GB/s is better):

| record | figure | source |
|---|---|---|
| naive fused kernel | "uncoalesced access pattern **leads to terrible performance**" | `dev/cuda/fused_residual_forward.cu` (comment on `fused_residual_forward2`) |
| fast fused kernels (warp-per-row, shared-mem staging) | benchmarked as `block_size … | time … ms | bandwidth … GB/s | elements: … ktok/ms` | `dev/cuda/fused_residual_forward.cu` harness, kernels 5/6 |
| development ladder | naive port (1) → **128-bit packed reads (2)** → coalesced warp-per-row fused (5/6) | top-of-file kernel ladder |

The `dev/cuda/fused_residual_forward.cu` benchmark emits `time`, `bandwidth`, and `ktok/ms` per block-size for
each kernel version at run time and checks correctness against the CPU reference; the repo records the
qualitative ordering (the naive fusion is "terrible," the warp-per-row shared-memory versions are the fast
ones) rather than a literal end-to-end tokens/sec delta for this fusion alone. The per-rung evidence is that
kernel ladder, and the structural argument — one full (B,T,C) read + one full (B,T,C) write of the residual sum
removed per seam. Bit-faithful (FP32-internal), so the 3.29 target is held.
