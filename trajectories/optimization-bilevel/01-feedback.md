Measured results — `baseline:t_rhg` (`is_final,true`), seeds {42, 123, 456} and mean.

## Toy convergence (lower steps / residual better)
| seed | convergence_steps | median_steps | final_residual | success_rate | runtime_sec |
|---|---|---|---|---|---|
| 42 | 261.256 | 303.0 | 0.030345 | 1.0 | 0.525684 |
| 123 | 260.457 | 303.0 | 0.029994 | 1.0 | 0.514880 |
| 456 | 260.423 | 301.0 | 0.029924 | 1.0 | 0.514734 |
| **mean** | **260.712** | **302.333** | **0.030088** | **1.0** | **0.518433** |

## Hyper-cleaning, linear (higher accuracy / f1 better)
| seed | test_accuracy | f1_score | cleaner_precision | cleaner_recall |
|---|---|---|---|---|
| 42 | 84.82 | 89.244934 | 0.826962 | 0.9692 |
| 123 | 84.10 | 89.085869 | 0.827444 | 0.9648 |
| 456 | 84.92 | 88.847656 | 0.828660 | 0.9576 |
| **mean** | **84.613** | **89.059486** | **0.827689** | **0.963867** |

## Hyper-cleaning, MLP (seed 42 only; higher better)
| seed | test_accuracy | f1_score | cleaner_precision | cleaner_recall |
|---|---|---|---|---|
| 42 | 84.79 | 89.348465 | 0.823549 | 0.9764 |
| **mean** | **84.79** | **89.348465** | **0.823549** | **0.9764** |
