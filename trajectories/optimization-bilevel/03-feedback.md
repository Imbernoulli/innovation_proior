Measured results — `baseline:g_pbgd` (`is_final,true`), seeds {42, 123, 456} and mean.

## Toy convergence (lower steps / residual better)
| seed | convergence_steps | median_steps | final_residual | success_rate | runtime_sec |
|---|---|---|---|---|---|
| 42 | 303.318 | 313.0 | 0.089126 | 1.0 | 0.674206 |
| 123 | 306.122 | 313.0 | 0.056776 | 1.0 | 0.694205 |
| 456 | 301.618 | 313.0 | 0.097786 | 1.0 | 0.673658 |
| **mean** | **303.686** | **313.0** | **0.081229** | **1.0** | **0.680690** |

## Hyper-cleaning, linear (higher accuracy / f1 better)
| seed | test_accuracy | f1_score | cleaner_precision | cleaner_recall |
|---|---|---|---|---|
| 42 | 89.77 | 83.521446 | 0.857564 | 0.8140 |
| 123 | 89.49 | 76.672234 | 0.802977 | 0.7336 |
| 456 | 90.25 | 81.692497 | 0.857520 | 0.7800 |
| **mean** | **89.837** | **80.628726** | **0.839354** | **0.775867** |

## Hyper-cleaning, MLP (seed 42 only; higher better)
| seed | test_accuracy | f1_score | cleaner_precision | cleaner_recall |
|---|---|---|---|---|
| 42 | 92.38 | 90.820122 | 0.889230 | 0.9280 |
| **mean** | **92.38** | **90.820122** | **0.889230** | **0.9280** |
