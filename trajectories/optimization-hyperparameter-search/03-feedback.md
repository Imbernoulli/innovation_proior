Measured results — `baseline:dehb` (`is_final,true`), seeds {42, 123, 456} and mean.

## XGBoost (budget 50)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | −0.388511 | 0.847548 | 80 |
| 123 | −0.399655 | 0.997118 | 116 |
| 456 | −0.416707 | 1.003067 | 116 |
| **mean** | **−0.401624** | **0.949244** | **104** |

## SVM (budget 40)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | 0.978932 | 0.973311 | 65 |
| 123 | 0.956097 | 0.982038 | 354 |
| 456 | 0.963127 | 0.988664 | 320 |
| **mean** | **0.966052** | **0.981338** | **246.33** |

## Neural Net (budget 40)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | −3086.08148 | 1.003472 | 70 |
| 123 | −3062.28388 | 1.007861 | 106 |
| 456 | −2997.87989 | 0.794007 | 99 |
| **mean** | **−3048.74842** | **0.935113** | **91.67** |
