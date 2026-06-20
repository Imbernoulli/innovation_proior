Measured result — record `2024-12-04_ValueEmbed`, log `00008ea0...txt`. Final line of the reproducible run
log. `val_loss` lower is better, ≤ 3.28; `train_time` lower is better.

| metric | value |
|---|---|
| val_loss | 3.2751 |
| train_time | 264,608 ms (≈ 4.41 min) |
| steps | 1530 |
| step_avg | 174.08 ms |

The dedicated per-layer token value embeddings cut the step count from 1750 to 1530 at the bar. The step
time went *up* slightly to 174.08 ms because of the extra twelve-times-wider embedding lookup, but the step
count dropped enough that the total wallclock still fell to ≈4.41 min. (World-record table record #14,
"Value Embeddings".)
