Measured results — `baseline:seag` (`is_final,true`). Lower is better. Seed 42 run deterministically;
the leaderboard repeats the identical row for seeds {42, 123, 456}.

## default-noise
| seed | final_gradient_norm | bilinear_fgn | delta_nu_fgn | auc_log_iter_log_grad |
|---|---|---|---|---|
| 42 | 0.135449 | 0.160590 | 0.110307 | −1.107236 |
| 123 | 0.135449 | 0.160590 | 0.110307 | −1.107236 |
| 456 | 0.135449 | 0.160590 | 0.110307 | −1.107236 |
| **mean** | **0.135449** | **0.160590** | **0.110307** | **−1.107236** |

## low-noise
| seed | final_gradient_norm | bilinear_fgn | delta_nu_fgn | auc_log_iter_log_grad |
|---|---|---|---|---|
| mean | 0.082633 | 0.158595 | 0.006672 | −1.547470 |

## high-noise
| seed | final_gradient_norm | bilinear_fgn | delta_nu_fgn | auc_log_iter_log_grad |
|---|---|---|---|---|
| mean | 0.380617 | 0.178678 | 0.582557 | −0.638068 |
