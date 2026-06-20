Measured result — record `2025-01-04_SoftCap`, log `31d6c427...txt`. Final line of the reproducible run
log. `val_loss` lower is better, ≤ 3.28; `train_time` lower is better.

| metric | value |
|---|---|
| val_loss | 3.2785 |
| train_time | 204,345 ms (≈ 3.4 min) |
| steps | 1390 |
| step_avg | 148.08 ms |

Lowering the cap from 30 to 15 let the step count drop from 1490 (the immediately prior configuration,
after intervening tuning) to 1390, pulling wallclock to ≈3.4 min. The record's README reports running the
new record 80 times: mean val loss 3.2791, std 0.0019, a one-sided t-test vs 3.28 giving p=0.0001 —
statistically below the bar. (World-record table record #18, "Lower logit softcap from 30 to 15".)
