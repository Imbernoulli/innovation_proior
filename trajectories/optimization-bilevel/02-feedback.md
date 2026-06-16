Measured results — `baseline:rhg` (`is_final,true`), seeds {42, 123, 456} and mean.

## Toy convergence (lower steps / residual better)
| seed | convergence_steps | median_steps | final_residual | success_rate | runtime_sec |
|---|---|---|---|---|---|
| 42 | 261.256 | 303.0 | 0.030345 | 1.0 | 0.534162 |
| 123 | 260.457 | 303.0 | 0.029994 | 1.0 | 0.513467 |
| 456 | 260.423 | 301.0 | 0.029924 | 1.0 | 0.539379 |
| **mean** | **260.712** | **302.333** | **0.030088** | **1.0** | **0.529003** |

## Hyper-cleaning, linear (higher accuracy / f1 better)
| seed | test_accuracy | f1_score | cleaner_precision | cleaner_recall |
|---|---|---|---|---|
| 42 | 84.82 | 89.534676 | 0.828737 | 0.9736 |
| 123 | 84.16 | 89.716179 | 0.831852 | 0.9736 |
| 456 | 84.92 | 89.389427 | 0.836003 | 0.9604 |
| **mean** | **84.633** | **89.546761** | **0.832197** | **0.9692** |

## Hyper-cleaning, MLP (seed 42 only; higher better)
| seed | test_accuracy | f1_score | cleaner_precision | cleaner_recall |
|---|---|---|---|---|
| 42 | 84.79 | 89.339172 | 0.821692 | 0.9788 |
| **mean** | **84.79** | **89.339172** | **0.821692** | **0.9788** |
