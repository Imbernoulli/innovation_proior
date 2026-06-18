Measured results — `baseline:mlp` (`is_final,true`), seed 42. One top-level seed, so the per-seed row is
the mean.

## test_accuracy by family
| seed | random-n30-s10-w4 | monotone-n40-s20-w4 | sparse-n60-s10-w4 |
|---|---|---|---|
| 42 | 0.9989 | 0.8461 | 0.9876 |
| **mean** | **0.9989** | **0.8461** | **0.9876** |

## aggregate
| metric | value |
|---|---|
| geometric mean (test_accuracy) | 0.9415 |
| base_rate_majority (random / monotone / sparse) | 0.5034 / 0.6277 / 0.5112 |
| mean_fit_seconds (random / monotone / sparse) | 3.94 / 3.91 / 3.91 |
