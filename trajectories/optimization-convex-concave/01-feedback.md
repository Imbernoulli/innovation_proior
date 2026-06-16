Measured results — `baseline:r_seg` (`is_final,true`). Lower is better. The harness runs seed 42
deterministically; the leaderboard repeats the identical row for seeds {42, 123, 456}.

## default-noise
| seed | final_gradient_norm | bilinear_fgn | delta_nu_fgn | auc_log_iter_log_grad |
|---|---|---|---|---|
| 42 | 0.751257 | 1.409909 | 0.092606 | −0.669446 |
| 123 | 0.751257 | 1.409909 | 0.092606 | −0.669446 |
| 456 | 0.751257 | 1.409909 | 0.092606 | −0.669446 |
| **mean** | **0.751257** | **1.409909** | **0.092606** | **−0.669446** |

## low-noise
| seed | final_gradient_norm | bilinear_fgn | delta_nu_fgn | auc_log_iter_log_grad |
|---|---|---|---|---|
| mean | 0.749993 | 1.407480 | 0.092506 | −0.675725 |

## high-noise
| seed | final_gradient_norm | bilinear_fgn | delta_nu_fgn | auc_log_iter_log_grad |
|---|---|---|---|---|
| mean | 0.774997 | 1.421081 | 0.128912 | −0.515712 |
