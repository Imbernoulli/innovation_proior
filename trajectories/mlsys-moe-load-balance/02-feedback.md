Measured results — `baseline:zigzag` (`is_final,true`), seed 42 (mean over 10 workload trials;
std is within-run trial variance).

| Config | balance | balance_node | locality | runtime_ms |
|---|---|---|---|---|
| deepseek-v3 | 0.659000 | 0.702209 | 1.000000 | 1.6472 |
| qwen3-moe   | 0.919030 | 0.946751 | 1.000000 | 0.8518 |
| deepseek-v2 | 0.905732 | 0.930966 | 1.000000 | 1.1591 |
| stress-skew | 0.221962 | 0.335967 | 1.000000 | 1.5613 |

| Config | balance_std | balance_node_std | locality_std | runtime_std |
|---|---|---|---|---|
| deepseek-v3 | 0.008598 | 0.009289 | 0.0 | 0.0627 |
| qwen3-moe   | 0.004606 | 0.005012 | 0.0 | 0.0716 |
| deepseek-v2 | 0.005396 | 0.004854 | 0.0 | 0.0737 |
| stress-skew | 0.001395 | 0.005892 | 0.0 | 0.0335 |

Task score (geometric mean across the four configs): **0.37500**.
