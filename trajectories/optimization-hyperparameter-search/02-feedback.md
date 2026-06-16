Measured results — `baseline:optuna_cma` (`is_final,true`), seeds {42, 123, 456} and mean.

## XGBoost (budget 50)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | −0.404518 | 0.801772 | 50 |
| 123 | −0.405250 | 0.645021 | 50 |
| 456 | −0.391865 | 0.764980 | 50 |
| **mean** | **−0.400544** | **0.737258** | **50** |

## SVM (budget 40)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | 0.978932 | 0.840799 | 40 |
| 123 | 0.978932 | 0.224080 | 40 |
| 456 | 0.975408 | 0.928333 | 40 |
| **mean** | **0.977757** | **0.664404** | **40** |

## Neural Net (budget 40)
| seed | best_val_score | convergence_auc | total_evals |
|---|---|---|---|
| 42 | −3066.20590 | 0.962500 | 40 |
| 123 | −3013.31892 | 0.881799 | 40 |
| 456 | −3021.75607 | 0.962500 | 40 |
| **mean** | **−3033.76030** | **0.935600** | **40** |
