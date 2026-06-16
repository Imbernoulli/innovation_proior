Measured results — `baseline:hyperband` (`is_final,true`), seeds {42, 123, 456} and mean.

## XGBoost (budget 50)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | −0.388511 | 0.916164 | 105 |
| 123 | −0.393145 | 0.976994 | 105 |
| 456 | −0.391865 | 0.987643 | 105 |
| **mean** | **−0.391174** | **0.960267** | **105** |

## SVM (budget 40)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | 0.978932 | 0.958140 | 95 |
| 123 | 0.978932 | 0.913752 | 95 |
| 456 | 0.975408 | 0.989892 | 95 |
| **mean** | **0.977757** | **0.953928** | **95** |

## Neural Net (budget 40)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | −3045.50206 | 0.995087 | 95 |
| 123 | −3070.29011 | 0.998160 | 95 |
| 456 | −3043.50752 | 0.855965 | 95 |
| **mean** | **−3053.09990** | **0.949737** | **95** |
