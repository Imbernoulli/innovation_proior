What this record changed: switched storage and the matmul/activation path from FP32 to **BF16** (`floatX =
__nv_bfloat16`, `CUBLAS_LOWP = CUDA_R_16BF`), with an **FP32 master-weight + FP32 moments** optimizer and
**stochastic rounding** on the cast-down to BF16. No loss scaler (BF16's exponent range removes the need).

Measured numbers (the repo's own figures; higher is better unless noted):

| quantity | value | source |
|---|---|---|
| Ampere (A100) datacenter peak used for MFU, BF16/TF32 | **312** TFLOPS (vs the FP32 lane) | `llmc/mfu.h` perf table (`AMPERE_DATACENTER = {156, 312, 312, 312, ...}`) |
| precision default | **BF16** (FP16 / FP32 selectable by compile flag) | `llmc/cublas_common.h`, `llmc/cuda_common.h` (`floatX`) |
| optimizer-state correctness device | FP32 master weights + FP32 `m`,`v`; **stochastic rounding** to BF16 | `llmc/adamw.cuh` (`adamw_update`, `stochastic_rounding`) |
| separate from FP32 path? | yes — `train_gpt2_fp32.cu` is the frozen FP32 reference; the mixed-precision mainline is `train_gpt2.cu` | repo file layout / README |

The repo splits the FP32 reference (`train_gpt2_fp32.cu`, "checkpointed early and frozen in time") from the
mixed-precision mainline (`train_gpt2.cu`) precisely because the BF16 path is the faster one; it does not
publish an isolated FP32-vs-BF16 end-to-end tokens/sec delta for the 124M run, so the per-rung evidence is the
peak-FLOPS table (the half-precision tensor-core lane the matmuls now run on) and the master-weight/stochastic-
rounding machinery that makes BF16 training hold the 3.29 target. The README also notes the finished mixed-
precision stack runs ~7% faster than PyTorch Nightly on the same model.
