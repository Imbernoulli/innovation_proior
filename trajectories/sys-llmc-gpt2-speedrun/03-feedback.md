What this record changed: introduced the `Packed128<T>` data structure forcing **128-bit (`LDG.128`/`STG.128`)
loads and stores** across the memory-bound kernels (GELU, residual, LayerNorm, AdamW), with streaming vs.
cached hints (`load128cs`/`store128`/`store128cg`). Each thread now moves an `int4`-wide packet (8 BF16 or 4
FP32) per memory instruction instead of one element.

Measured numbers (the repo's own per-kernel benchmark instrumentation; lower time / higher GB/s is better):

| kernel file | what it benchmarks | reported metrics (per launch config) |
|---|---|---|
| `dev/cuda/gelu_forward.cu` | kernel 1 (naive scalar) vs **kernel 2 (BF16 + Packed128)** | `block_size … | time … ms | bandwidth … GB/s` |
| `dev/cuda/fused_residual_forward.cu` | "version 2 **packs input into 128 bit memory reads**" | `block_size … | time … ms | bandwidth … GB/s | elements: … ktok/ms` |
| `dev/cuda/layernorm_forward.cu` | Packed128 row loads, FP32 reductions | `block_size … | time … ms | bandwidth … GB/s` |

Each `dev/cuda/*.cu` benchmark harness prints `time … ms` and effective `bandwidth … GB/s` per block-size for
every kernel version and checks each against the CPU reference for correctness; the Packed128 versions are the
later (faster, higher-bandwidth) entries in each file's kernel ladder. The repo's per-kernel numbers are
emitted at run time by these harnesses rather than recorded as literals in source, and there is no isolated
end-to-end tokens/sec delta attributed to vectorization alone; the per-rung evidence is the kernel ladder
itself (naive scalar → Packed128 128-bit), which is the documented mechanism by which these bandwidth-bound
kernels approach the HBM roofline. The change is bit-exact (arithmetic unchanged), so the 3.29 target is held.
