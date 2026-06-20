Measured result — record `2024-11-06_ShortcutsTweaks` (the cumulative +tanh-softcap reproducible log,
dd7304a6…txt). Final line of the run log. `val_loss` lower is better, ≤ 3.28; `train_time` lower is better.

| metric | value |
|---|---|
| val_loss | 3.2791 |
| train_time | 493,742 ms (≈ 8.2 min) |
| steps | 3200 |
| step_avg | 154.78 ms |

Added incrementally on top of the 10.8-min record, the four shortcuts — value residual with a learnable
per-layer lambda, the x0 embed shortcut, Muon momentum warmup, and the tanh logit softcap — cut the step
count to 3200 (from 4578) while holding the bar, dropping the wallclock to ~8.2 min. (Record #9, "Value and
embedding skip connections, momentum warmup, logit softcap".)
