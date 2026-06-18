Measured results — `baseline:gbdt` (`is_final,true`), seed 42. One top-level seed, so the per-seed row is
the mean.

## test_accuracy by family
| seed | random-n30-s10-w4 | monotone-n40-s20-w4 | sparse-n60-s10-w4 |
|---|---|---|---|
| 42 | 1.0000 | 0.9996 | 1.0000 |
| **mean** | **1.0000** | **0.9996** | **1.0000** |

## aggregate
| metric | value |
|---|---|
| geometric mean (test_accuracy) | 0.9999 |
| base_rate_majority (random / monotone / sparse) | 0.5034 / 0.6277 / 0.5112 |
| mean_fit_seconds (random / monotone / sparse) | 12.86 / 16.58 / 24.01 |
