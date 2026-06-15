Measured results — `baseline:flash_v3` (`is_final,true`), seed {42} (the harness runs seed 42 only; mean = the single seed).

## hdim64_seq4k (batch 4, seqlen 4096, heads 32, headdim 64)
| seed | tflops | latency_ms | max_diff | correct | speedup_vs_sdpa |
|---|---|---|---|---|---|
| 42 | 332.1193 | 0.8276 | 0.001953125 | 1 | 1.1584 |
| **mean** | **332.1193** | **0.8276** | **0.001953125** | **1** | **1.1584** |

## hdim128_seq8k (batch 2, seqlen 8192, heads 16, headdim 128)
| seed | tflops | latency_ms | max_diff | correct | speedup_vs_sdpa |
|---|---|---|---|---|---|
| 42 | 406.4221 | 1.3527 | 0.001953125 | 1 | 1.2972 |
| **mean** | **406.4221** | **1.3527** | **0.001953125** | **1** | **1.2972** |

## hdim256_seq16k (batch 1, seqlen 16384, heads 8, headdim 256)
| seed | tflops | latency_ms | max_diff | correct | speedup_vs_sdpa |
|---|---|---|---|---|---|
| 42 | 240.5420 | 4.5710 | 0.001953125 | 1 | 0.8360 |
| **mean** | **240.5420** | **4.5710** | **0.001953125** | **1** | **0.8360** |
