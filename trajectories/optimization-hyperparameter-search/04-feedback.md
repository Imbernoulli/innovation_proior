Measured results — `baseline:tpe` (`is_final,true`), seeds {42, 123, 456} and mean.

## XGBoost (budget 50)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | −0.391354 | 0.913637 | 50 |
| 123 | −0.390420 | 0.948581 | 50 |
| 456 | −0.394426 | 0.936397 | 50 |
| **mean** | **−0.392067** | **0.932872** | **50** |

## SVM (budget 40)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | 0.978932 | 0.922994 | 40 |
| 123 | 0.978932 | 0.793386 | 40 |
| 456 | 0.980686 | 0.912390 | 40 |
| **mean** | **0.979517** | **0.876257** | **40** |

## Neural Net (budget 40)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | −3063.60619 | 0.958000 | 40 |
| 123 | −3041.41230 | 0.797801 | 40 |
| 456 | −3039.38202 | 0.736065 | 40 |
| **mean** | **−3048.13351** | **0.830621** | **40** |
