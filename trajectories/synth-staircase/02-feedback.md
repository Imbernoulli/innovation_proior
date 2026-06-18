Measured results — `baseline:mean_field_sgd` (`is_final,true`), seeds {42, 123, 456} and mean.

## h1 (leap-1 staircase)
| seed | test_mse | fourier_recovery | score |
|---|---|---|---|
| 42 | 2.431203 | 0.883939 | 0.087931 |
| 123 | 2.480340 | 0.894372 | 0.083715 |
| 456 | 2.433240 | 0.879936 | 0.087752 |
| **mean** | **2.448261** | **0.886082** | **0.086466** |

## h2 (leap-2 non-MSP chain)
| seed | test_mse | fourier_recovery | score |
|---|---|---|---|
| 42 | 2.960399 | 0.986299 | 0.051798 |
| 123 | 3.034799 | 1.011238 | 0.048084 |
| 456 | 2.991479 | 0.996699 | 0.050213 |
| **mean** | **2.995559** | **0.998079** | **0.050032** |

## h3 (leap-3 monomial)
| seed | test_mse | fourier_recovery | score |
|---|---|---|---|
| 42 | 1.000052 | 0.999219 | 0.367860 |
| 123 | 1.001326 | 0.999886 | 0.367392 |
| 456 | 1.000471 | 0.999491 | 0.367706 |
| **mean** | **1.000616** | **0.999532** | **0.367653** |

Aggregate (geometric mean of per-env `score` means): **0.1167**.
