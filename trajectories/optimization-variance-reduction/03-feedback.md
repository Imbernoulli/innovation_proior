Measured results — `baseline:storm_plus` (`is_final,true`), seeds {42, 123, 456} and mean.

## logistic (accuracy, higher is better)
| seed | best_test_accuracy | final_test_accuracy |
|---|---|---|
| 42 | 92.62 | 92.54 |
| 123 | 92.47 | 92.46 |
| 456 | 92.43 | 92.43 |
| **mean** | **92.507** | **92.477** |

## mlp (accuracy, higher is better)
| seed | best_test_accuracy | final_test_accuracy |
|---|---|---|
| 42 | 54.17 | 52.69 |
| 123 | 54.04 | 52.92 |
| 456 | 54.38 | 51.89 |
| **mean** | **54.197** | **52.50** |

## conditioned (MSE, lower is better)
| seed | best_test_mse | final_test_mse |
|---|---|---|
| 42 | 0.014749 | 0.019557 |
| 123 | 0.016263 | 0.017197 |
| 456 | 0.014302 | 0.017396 |
| **mean** | **0.015105** | **0.018050** |
