Measured results — `baseline:squat_subspace_4bit` (`is_final,true`), seed `42` (the only seed; mean equals it).

## Quality (`final_score`, 0-100)
| seed | hotpotqa | passage_retrieval | repobench | niah | gsm8k |
|---|---|---|---|---|---|
| 42 | 36.532065 | 59.964286 | 46.48 | 65.306122 | 31.76649 |
| **mean** | **36.532065** | **59.964286** | **46.48** | **65.306122** | **31.76649** |

## Efficiency (`effective_kv_bits` / `kv_compression_ratio`)
| seed | metric | hotpotqa | passage_retrieval | repobench | niah | gsm8k |
|---|---|---|---|---|---|---|
| 42 | effective_kv_bits | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 |
| 42 | kv_compression_ratio | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 |

## Runtime (`runtime_seconds`, diagnostic, not scored)
| seed | hotpotqa | passage_retrieval | repobench | niah | gsm8k |
|---|---|---|---|---|---|
| 42 | 1392.7 | 1348.5 | 2742.0 | 1563.3 | 29152.6 |
