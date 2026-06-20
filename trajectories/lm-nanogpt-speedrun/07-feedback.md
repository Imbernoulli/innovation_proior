Measured result — record `2024-11-24_WindowWarmup` (log `cf9e4571…txt`). Final line of the reproducible run
log. `val_loss` lower is better, ≤ 3.28; `train_time` lower is better.

| metric | value |
|---|---|
| val_loss | 3.2759 |
| train_time | 279,948 ms (≈ 4.66 min) |
| steps | 1750 |
| step_avg | 160.89 ms |

Growing the window linearly from 64 to ~1792 tokens makes early steps cheaper and acts as a
short-to-long-context curriculum: the step count drops to 1750 (from 1875) at a slightly lower per-step time,
cutting the wallclock to ~4.66 min, and at a better and steadier val_loss (3.2759) than the fixed-window
record. (Record #13, "Attention window warmup".)
