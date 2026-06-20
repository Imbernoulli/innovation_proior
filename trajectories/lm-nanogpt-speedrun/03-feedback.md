Measured result — record `2024-11-03_UntieEmbed`. Final line of the reproducible run log. `val_loss` lower
is better, ≤ 3.28; `train_time` lower is better.

| metric | value |
|---|---|
| val_loss | 3.2762 |
| train_time | 648,063 ms (≈ 10.8 min) |
| steps | 4578 |
| step_avg | 141.87 ms |

Untying the embedding/head, zero-initializing the head, and normalizing the embedding after lookup cut the
step count to 4578 (from 5100) at the bar, dropping the wallclock to ~10.8 min. The record's note records
the trade explicitly: untying adds 39M parameters but leaves active parameters and inference throughput
unchanged. (Record #8, "Untied embedding and head".)
