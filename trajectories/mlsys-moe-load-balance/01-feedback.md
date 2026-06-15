Measured results — `baseline:greedy` (`is_final,true`), seed 42 (mean over 10 workload trials;
std is within-run trial variance).

| Config | balance | balance_node | locality | runtime_ms |
|---|---|---|---|---|
| deepseek-v3 | 0.678627 | 0.702209 | 1.000000 | 248.5075 |
| qwen3-moe   | 0.940328 | 0.946751 | 1.000000 | 102.3190 |
| deepseek-v2 | 0.925448 | 0.930967 | 1.000000 | 153.3683 |
| stress-skew | 0.221977 | 0.335967 | 1.000000 | 256.5723 |

| Config | balance_std | balance_node_std | locality_std | runtime_std |
|---|---|---|---|---|
| deepseek-v3 | 0.008577 | 0.009289 | 0.0 | 1.6236 |
| qwen3-moe   | 0.005041 | 0.005012 | 0.0 | 0.5916 |
| deepseek-v2 | 0.004879 | 0.004854 | 0.0 | 1.5132 |
| stress-skew | 0.001398 | 0.005892 | 0.0 | 1.1844 |

Task score (geometric mean across the four configs): **0.25531**.
