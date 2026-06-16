Measured results — `baseline:svrg` (`is_final,true`), seeds {42, 123, 456} and mean.

## logistic (accuracy, higher is better)
| seed | best_test_accuracy | final_test_accuracy |
|---|---|---|
| 42 | 92.58 | 92.58 |
| 123 | 92.66 | 92.66 |
| 456 | 92.65 | 92.61 |
| **mean** | **92.63** | **92.617** |

## mlp (accuracy, higher is better)
| seed | best_test_accuracy | final_test_accuracy |
|---|---|---|
| 42 | 53.44 | 49.72 |
| 123 | 52.51 | 48.83 |
| 456 | 52.06 | 51.71 |
| **mean** | **52.67** | **50.087** |

## conditioned (MSE, lower is better)
| seed | best_test_mse | final_test_mse |
|---|---|---|
| 42 | 2582.674063 | 85157854838.784 |
| 123 | 1017.325406 | nan |
| 456 | 1252.079094 | nan |
| **mean** | **1617.359521** | **85157854838.784** |
