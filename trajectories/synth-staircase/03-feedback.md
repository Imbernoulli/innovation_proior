Measured results — `baseline:curriculum_layerwise` (`is_final,true`), seeds {42, 123, 456} and mean.

## h1 (leap-1 staircase)
| seed | test_mse | fourier_recovery | score |
|---|---|---|---|
| 42 | 0.469305 | 0.324915 | 0.625437 |
| 123 | 0.460783 | 0.323251 | 0.630790 |
| 456 | 0.459798 | 0.327509 | 0.631411 |
| **mean** | **0.463295** | **0.325225** | **0.629213** |

## h2 (leap-2 non-MSP chain)
| seed | test_mse | fourier_recovery | score |
|---|---|---|---|
| 42 | 0.978321 | 0.530963 | 0.375942 |
| 123 | 1.033402 | 0.553062 | 0.355794 |
| 456 | 0.980105 | 0.536057 | 0.375272 |
| **mean** | **0.997276** | **0.540027** | **0.369003** |

## h3 (leap-3 monomial)
| seed | test_mse | fourier_recovery | score |
|---|---|---|---|
| 42 | 0.015073 | 0.116411 | 0.985040 |
| 123 | 0.020387 | 0.131951 | 0.979820 |
| 456 | 0.018889 | 0.128862 | 0.981288 |
| **mean** | **0.018116** | **0.125741** | **0.982049** |

Aggregate (geometric mean of per-env `score` means): **0.6109**.
