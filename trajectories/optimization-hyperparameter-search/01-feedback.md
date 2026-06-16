Measured results — `baseline:random_search` (`is_final,true`), seeds {42, 123, 456} and mean.

## XGBoost (budget 50)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | −0.396658 | 0.957041 | 50 |
| 123 | −0.393145 | 0.962503 | 50 |
| 456 | −0.391865 | 0.917713 | 50 |
| **mean** | **−0.393889** | **0.945752** | **50** |

## SVM (budget 40)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | 0.978932 | 0.889855 | 40 |
| 123 | 0.978932 | 0.562513 | 40 |
| 456 | 0.975408 | 0.913161 | 40 |
| **mean** | **0.977757** | **0.788510** | **40** |

## Neural Net (budget 40)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | −3070.18992 | 0.961907 | 40 |
| 123 | −3013.31892 | 0.664774 | 40 |
| 456 | −3067.41709 | 0.690738 | 40 |
| **mean** | **−3050.30864** | **0.772473** | **40** |
