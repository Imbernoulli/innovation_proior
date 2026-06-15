Measured results — `baseline:flash_v2` (`is_final,true`), seed {42} (the harness runs seed 42 only; mean = the single seed).

## hdim64_seq4k (batch 4, seqlen 4096, heads 32, headdim 64)
| seed | tflops | latency_ms | max_diff | correct | speedup_vs_sdpa |
|---|---|---|---|---|---|
| 42 | 308.2918 | 0.8916 | 0.001953125 | 1 | 1.0893 |
| **mean** | **308.2918** | **0.8916** | **0.001953125** | **1** | **1.0893** |

## hdim128_seq8k (batch 2, seqlen 8192, heads 16, headdim 128)
| seed | tflops | latency_ms | max_diff | correct | speedup_vs_sdpa |
|---|---|---|---|---|---|
| 42 | 200.1523 | 2.7467 | 0.001953125 | 1 | 0.6377 |
| **mean** | **200.1523** | **2.7467** | **0.001953125** | **1** | **0.6377** |

## hdim256_seq16k (batch 1, seqlen 16384, heads 8, headdim 256)
| seed | tflops | latency_ms | max_diff | correct | speedup_vs_sdpa |
|---|---|---|---|---|---|
| 42 | 240.3888 | 4.5739 | 0.001953125 | 1 | 0.8326 |
| **mean** | **240.3888** | **4.5739** | **0.001953125** | **1** | **0.8326** |
