Measured results — `baseline:storm` (`is_final,true`), seeds {42, 123, 456} and mean.

## logistic (accuracy, higher is better)
| seed | best_test_accuracy | final_test_accuracy |
|---|---|---|
| 42 | 92.55 | 92.35 |
| 123 | 92.43 | 92.33 |
| 456 | 92.51 | 92.39 |
| **mean** | **92.497** | **92.357** |

## mlp (accuracy, higher is better)
| seed | best_test_accuracy | final_test_accuracy |
|---|---|---|
| 42 | 53.67 | 52.56 |
| 123 | 53.78 | 52.53 |
| 456 | 54.28 | 52.38 |
| **mean** | **53.91** | **52.49** |

## conditioned (MSE, lower is better)
| seed | best_test_mse | final_test_mse |
|---|---|---|
| 42 | 3.505524 | 3.908639 |
| 123 | 0.849033 | 4.064009 |
| 456 | 0.912248 | 0.912248 |
| **mean** | **1.755602** | **2.961632** |
