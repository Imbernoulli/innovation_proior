Measured results — `baseline:kvtuner4_pertoken_qwen25_3b` (`is_final,true`), seed `42` (the only seed; mean equals it).

## Quality (`final_score`, 0-100)
| seed | hotpotqa | passage_retrieval | repobench | niah | gsm8k |
|---|---|---|---|---|---|
| 42 | 39.470331 | 56.633333 | 43.743333 | 53.741497 | 43.517817 |
| **mean** | **39.470331** | **56.633333** | **43.743333** | **53.741497** | **43.517817** |

## Efficiency (`effective_kv_bits` / `kv_compression_ratio`)
| seed | metric | hotpotqa | passage_retrieval | repobench | niah | gsm8k |
|---|---|---|---|---|---|---|
| 42 | effective_kv_bits | 3.638889 | 3.638889 | 3.638889 | 3.638889 | 3.638889 |
| 42 | kv_compression_ratio | 4.396947 | 4.396947 | 4.396947 | 4.396947 | 4.396947 |

## Runtime (`runtime_seconds`, diagnostic, not scored)
| seed | hotpotqa | passage_retrieval | repobench | niah | gsm8k |
|---|---|---|---|---|---|
| 42 | 514.2 | 469.8 | 1121.2 | 588.9 | 12877.6 |
