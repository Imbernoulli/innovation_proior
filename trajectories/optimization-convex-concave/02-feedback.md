Measured results — `baseline:seg` (`is_final,true`). Lower is better. Seed 42 run deterministically;
the leaderboard repeats the identical row for seeds {42, 123, 456}.

## default-noise
| seed | final_gradient_norm | bilinear_fgn | delta_nu_fgn | auc_log_iter_log_grad |
|---|---|---|---|---|
| 42 | 0.182141 | 0.173788 | 0.190493 | −0.346938 |
| 123 | 0.182141 | 0.173788 | 0.190493 | −0.346938 |
| 456 | 0.182141 | 0.173788 | 0.190493 | −0.346938 |
| **mean** | **0.182141** | **0.173788** | **0.190493** | **−0.346938** |

## low-noise
| seed | final_gradient_norm | bilinear_fgn | delta_nu_fgn | auc_log_iter_log_grad |
|---|---|---|---|---|
| mean | 0.116935 | 0.162024 | 0.071846 | −0.442062 |

## high-noise
| seed | final_gradient_norm | bilinear_fgn | delta_nu_fgn | auc_log_iter_log_grad |
|---|---|---|---|---|
| mean | 0.582105 | 0.227585 | 0.936626 | 0.076484 |
