Measured results — `baseline:flash_v1` (`is_final,true`), seed {42} (the harness runs seed 42 only; mean = the single seed).

## hdim64_seq4k (batch 4, seqlen 4096, heads 32, headdim 64)
| seed | tflops | latency_ms | max_diff | correct | speedup_vs_sdpa |
|---|---|---|---|---|---|
| 42 | 269.759 | 1.019 | 0.0009765625 | 1 | 0.9449 |
| **mean** | **269.759** | **1.019** | **0.0009765625** | **1** | **0.9449** |

## hdim128_seq8k (batch 2, seqlen 8192, heads 16, headdim 128)
| seed | tflops | latency_ms | max_diff | correct | speedup_vs_sdpa |
|---|---|---|---|---|---|
| 42 | 297.3736 | 1.8487 | 0.0004882812 | 1 | 0.9420 |
| **mean** | **297.3736** | **1.8487** | **0.0004882812** | **1** | **0.9420** |

## hdim256_seq16k (batch 1, seqlen 16384, heads 8, headdim 256)
| seed | tflops | latency_ms | max_diff | correct | speedup_vs_sdpa |
|---|---|---|---|---|---|
| 42 | 234.2001 | 4.6948 | 0.0004882812 | 1 | 0.8079 |
| **mean** | **234.2001** | **4.6948** | **0.0004882812** | **1** | **0.8079** |
