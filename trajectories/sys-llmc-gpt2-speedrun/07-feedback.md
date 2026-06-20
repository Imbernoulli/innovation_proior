What this record changed: fused the MLP's GELU (and bias) into the cuBLASLt GEMM epilogue instead of running a
standalone GELU kernel — forward via `CUBLASLT_EPILOGUE_GELU_AUX_BIAS` (writes the post-GELU output plus the
pre-GELU values to an auxiliary buffer for backward), backward via `CUBLASLT_EPILOGUE_DGELU` (folds the GELU
derivative into the backward matmul). Eliminates the HBM round-trip of the (B·T·4C) pre-GELU tensor, the
block's largest activation.

Measured / documented numbers (the repo's own statements):

| record | figure | source |
|---|---|---|
| design rationale for cuBLASLt over cuBLAS | "making cuBLAS for matmuls the default … is a no-brainer … a single line of interpretable code" — and the cuBLASLt epilogue absorbs bias/GELU fusion | `README.md` ("repo" philosophy section) |
| forward fusion epilogue | `CUBLASLT_EPILOGUE_GELU_AUX_BIAS` (bias→GELU + saved pre-GELU aux) | `llmc/matmul.cuh` (`matmul_cublaslt`) |
| backward fusion epilogue | `CUBLASLT_EPILOGUE_DGELU` (GELU gradient fused) | `llmc/matmul.cuh` |
| GELU definition (unchanged math) | tanh-approx GELU, `GELU_SCALING_FACTOR = sqrtf(2/π)` | `dev/cuda/gelu_forward.cu`, `llmc/gelu.cuh` |

The fused-GELU epilogue is the production matmul path in the mainline trainer (`matmul_cublaslt` is called with
a `pre_gelu` pointer for the MLP up-projection and with `backward=true` for the corresponding backward GEMM).
The repo does not publish an isolated end-to-end tokens/sec delta for toggling GELU fusion alone; the per-rung
evidence is the epilogue wiring itself and the structural argument — a standalone GELU/dGELU kernel reading and
writing the (B·T·4C) tensor (~3 GB BF16) twice per layer per step is replaced by free in-epilogue application.
The GELU is the same tanh-approximation applied at the same point, so the change is correct to tolerance and the
3.29 target is held.
