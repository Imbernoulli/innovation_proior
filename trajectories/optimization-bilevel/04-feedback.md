Measured results — `baseline:v_pbgd` (`is_final,true`), seeds {42, 123, 456} and mean.

## Toy convergence (lower steps / residual better)
| seed | convergence_steps | median_steps | final_residual | success_rate | runtime_sec |
|---|---|---|---|---|---|
| 42 | 261.256 | 303.0 | 0.030345 | 1.0 | 0.502807 |
| 123 | 260.457 | 303.0 | 0.029994 | 1.0 | 0.516588 |
| 456 | 260.423 | 301.0 | 0.029924 | 1.0 | 0.517659 |
| **mean** | **260.712** | **302.333** | **0.030088** | **1.0** | **0.512351** |

## Hyper-cleaning, linear (higher accuracy / f1 better)
| seed | test_accuracy | f1_score | cleaner_precision | cleaner_recall |
|---|---|---|---|---|
| 42 | 90.08 | 91.810837 | 0.883790 | 0.9552 |
| 123 | 89.88 | 91.957573 | 0.887896 | 0.9536 |
| 456 | 90.33 | 91.396599 | 0.882638 | 0.9476 |
| **mean** | **90.097** | **91.721670** | **0.884775** | **0.952133** |

## Hyper-cleaning, MLP (seed 42 only; higher better)
| seed | test_accuracy | f1_score | cleaner_precision | cleaner_recall |
|---|---|---|---|---|
| 42 | 91.48 | 92.050041 | 0.887199 | 0.9564 |
| **mean** | **91.48** | **92.050041** | **0.887199** | **0.9564** |
