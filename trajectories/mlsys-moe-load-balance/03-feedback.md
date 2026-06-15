Measured results — `baseline:flat_zigzag` (`is_final,true`), seed 42 (mean over 10 workload trials;
std is within-run trial variance).

| Config | balance | balance_node | locality | runtime_ms |
|---|---|---|---|---|
| deepseek-v3 | 0.980332 | 0.984442 | 0.915379 | 4.8350 |
| qwen3-moe   | 0.974809 | 0.983491 | 0.960669 | 1.6160 |
| deepseek-v2 | 0.983199 | 0.988721 | 0.966865 | 2.0559 |
| stress-skew | 0.931268 | 0.938060 | 0.727849 | 7.0557 |

| Config | balance_std | balance_node_std | locality_std | runtime_std |
|---|---|---|---|---|
| deepseek-v3 | 0.000465 | 0.000316 | 0.001367 | 0.4237 |
| qwen3-moe   | 0.000700 | 0.000444 | 0.002191 | 0.0321 |
| deepseek-v2 | 0.000399 | 0.000335 | 0.001948 | 0.0791 |
| stress-skew | 0.001326 | 0.001153 | 0.001566 | 0.0783 |

Task score (geometric mean across the four configs): **0.38204**.
