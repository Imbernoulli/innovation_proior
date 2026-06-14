Measured results — `baseline:kvtuner4_kivi_qwen25_3b` (`is_final,true`), seed `42` (the only seed; mean equals it).

## Quality (`final_score`, 0-100)
| seed | hotpotqa | passage_retrieval | repobench | niah | gsm8k |
|---|---|---|---|---|---|
| 42 | 35.77662 | 57.111111 | 46.02 | 64.62585 | 34.723275 |
| **mean** | **35.77662** | **57.111111** | **46.02** | **64.62585** | **34.723275** |

## Efficiency (`effective_kv_bits` / `kv_compression_ratio`)
| seed | metric | hotpotqa | passage_retrieval | repobench | niah | gsm8k |
|---|---|---|---|---|---|---|
| 42 | effective_kv_bits | 3.166667 | 3.166667 | 3.166667 | 3.166667 | 3.166667 |
| 42 | kv_compression_ratio | 5.052632 | 5.052632 | 5.052632 | 5.052632 | 5.052632 |

## Runtime (`runtime_seconds`, diagnostic, not scored)
| seed | hotpotqa | passage_retrieval | repobench | niah | gsm8k |
|---|---|---|---|---|---|
| 42 | 580.3 | 514.2 | 1294.4 | 600.9 | 12658.3 |
