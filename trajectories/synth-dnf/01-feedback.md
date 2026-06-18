Measured results — `baseline:deep_dnf` (`is_final,true`), seed 42. One top-level seed, so the per-seed
row is the mean.

## test_accuracy by family
| seed | random-n30-s10-w4 | monotone-n40-s20-w4 | sparse-n60-s10-w4 |
|---|---|---|---|
| 42 | 0.7605 | 0.9088 | 0.8986 |
| **mean** | **0.7605** | **0.9088** | **0.8986** |

## aggregate
| metric | value |
|---|---|
| geometric mean (test_accuracy) | 0.8532 |
| base_rate_majority (random / monotone / sparse) | 0.5034 / 0.6277 / 0.5112 |
| mean_fit_seconds (random / monotone / sparse) | 12.63 / 12.29 / 37.97 |
