Measured results — `baseline:rain` (`is_final,true`). Lower is better. Seed 42 run deterministically;
the leaderboard repeats the identical row for seeds {42, 123, 456}.

## default-noise
| seed | final_gradient_norm | bilinear_fgn | delta_nu_fgn | auc_log_iter_log_grad |
|---|---|---|---|---|
| 42 | 0.021594 | 0.020523 | 0.022665 | −0.797146 |
| 123 | 0.021594 | 0.020523 | 0.022665 | −0.797146 |
| 456 | 0.021594 | 0.020523 | 0.022665 | −0.797146 |
| **mean** | **0.021594** | **0.020523** | **0.022665** | **−0.797146** |

## low-noise
| seed | final_gradient_norm | bilinear_fgn | delta_nu_fgn | auc_log_iter_log_grad |
|---|---|---|---|---|
| mean | 0.008648 | 0.014857 | 0.002439 | −1.086840 |

## high-noise
| seed | final_gradient_norm | bilinear_fgn | delta_nu_fgn | auc_log_iter_log_grad |
|---|---|---|---|---|
| mean | 0.080775 | 0.047147 | 0.114402 | −0.320045 |
