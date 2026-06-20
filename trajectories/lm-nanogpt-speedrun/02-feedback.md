Measured result — record `2024-10-14_ModernArch`. Final line of the reproducible run log. `val_loss`
lower is better, ≤ 3.28; `train_time` lower is better.

| metric | value |
|---|---|
| val_loss | 3.2741 |
| train_time | 910,546 ms (≈ 15.2 min) |
| steps | 5100 |
| step_avg | 178.89 ms |

The modernized block reaches the bar in 5100 steps (down from 6200) and lower per-step time (178.89 ms vs
216.33 ms), for a large wallclock cut to ~15.2 min — and at a better val loss (3.2741), confirming the new
architecture both fits faster and runs cheaper per step. (Record #5 in the repository's world-record
table, "Pad embeddings, ReLU², zero-init projections, QK-norm".)
