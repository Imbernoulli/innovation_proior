Measured results — `baseline:streamingllm` (`is_final,true`), seed 42 (the single seed; mean equals it).

## final_score (0-100, higher better)
| seed | hotpotqa | passage_retrieval | repobench | longbench_v2 | gsm8k |
|---|---|---|---|---|---|
| 42 | 25.586 | 53.111 | 43.153 | 29.622 | 1.744 |
| **mean** | **25.586** | **53.111** | **43.153** | **29.622** | **1.744** |

## mean_retained_fraction (lower better)
| seed | hotpotqa | passage_retrieval | repobench | longbench_v2 | gsm8k |
|---|---|---|---|---|---|
| 42 | 0.1999 | 0.1999 | 0.1999 | 0.2000 | 0.1932 |
| **mean** | **0.1999** | **0.1999** | **0.1999** | **0.2000** | **0.1932** |

## runtime_seconds (lower better)
| seed | hotpotqa | passage_retrieval | repobench | longbench_v2 | gsm8k |
|---|---|---|---|---|---|
| 42 | 360.9 | 330.9 | 671.7 | 2592.9 | 7110.3 |
| **mean** | **360.9** | **330.9** | **671.7** | **2592.9** | **7110.3** |
