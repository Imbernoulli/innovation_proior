Measured results — `baseline:kivi_overlap_4bit` (`is_final,true`), seed `42` (the only seed; mean equals it).

## Quality (`final_score`, 0-100)
| seed | hotpotqa | passage_retrieval | repobench | niah | gsm8k |
|---|---|---|---|---|---|
| 42 | 37.684238 | 61.083333 | 47.71 | 65.306122 | 31.842305 |
| **mean** | **37.684238** | **61.083333** | **47.71** | **65.306122** | **31.842305** |

## Efficiency (`effective_kv_bits` / `kv_compression_ratio`)
| seed | metric | hotpotqa | passage_retrieval | repobench | niah | gsm8k |
|---|---|---|---|---|---|---|
| 42 | effective_kv_bits | 4.1875 | 4.1875 | 4.1875 | 4.1875 | 4.1875 |
| 42 | kv_compression_ratio | 3.820896 | 3.820896 | 3.820896 | 3.820896 | 3.820896 |

## Runtime (`runtime_seconds`, diagnostic, not scored)
| seed | hotpotqa | passage_retrieval | repobench | niah | gsm8k |
|---|---|---|---|---|---|
| 42 | 830.2 | 747.9 | 1970.7 | 732.4 | 13460.6 |
