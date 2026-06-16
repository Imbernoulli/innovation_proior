Measured results — `baseline:bohb` (`is_final,true`), seeds {42, 123, 456} and mean.

## XGBoost (budget 50)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | −0.388511 | 0.916164 | 105 |
| 123 | −0.392800 | 0.976698 | 105 |
| 456 | −0.385948 | 0.982249 | 105 |
| **mean** | **−0.389086** | **0.958370** | **105** |

## SVM (budget 40)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | 0.978932 | 0.958140 | 95 |
| 123 | 0.978932 | 0.913752 | 95 |
| 456 | 0.980686 | 0.977407 | 95 |
| **mean** | **0.979517** | **0.949766** | **95** |

## Neural Net (budget 40)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | −3031.23504 | 0.987420 | 95 |
| 123 | −3019.66786 | 0.996873 | 95 |
| 456 | −2993.78102 | 0.567070 | 95 |
| **mean** | **−3014.89464** | **0.850454** | **95** |
