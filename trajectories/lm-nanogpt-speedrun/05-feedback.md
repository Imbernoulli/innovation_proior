Measured result — record `2024-11-10_UNetDoubleLr` (log c87bb826…txt). Final line of the reproducible run
log. `val_loss` lower is better, ≤ 3.28; `train_time` lower is better.

| metric | value |
|---|---|
| val_loss | 3.2753 |
| train_time | 433,570 ms (≈ 7.2 min) |
| steps | 3000 |
| step_avg | 145.01 ms |

The symmetric U-net skip connections over depth, plus the doubled learning rate the cleaner gradient flow
allows, cut the step count to 3000 (from 3200) and the step time as well, dropping the wallclock to ~7.2 min
while improving the val_loss to 3.2753. (Record #11, "U-net pattern skip connections & double lr".)
