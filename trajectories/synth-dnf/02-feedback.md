Measured results — `baseline:rf` (`is_final,true`), seed 42. One top-level seed, so the per-seed row is
the mean.

## test_accuracy by family
| seed | random-n30-s10-w4 | monotone-n40-s20-w4 | sparse-n60-s10-w4 |
|---|---|---|---|
| 42 | 0.9346 | 0.8536 | 0.9312 |
| **mean** | **0.9346** | **0.8536** | **0.9312** |

## aggregate
| metric | value |
|---|---|
| geometric mean (test_accuracy) | 0.9057 |
| base_rate_majority (random / monotone / sparse) | 0.5034 / 0.6277 / 0.5112 |
| mean_fit_seconds (random / monotone / sparse) | 0.67 / 0.68 / 0.68 |
