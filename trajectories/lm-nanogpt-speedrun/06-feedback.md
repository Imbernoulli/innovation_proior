Measured result — record `2024-11-19_FlexAttention` (log `8384493d…txt`). Final line of the reproducible run
log. `val_loss` lower is better, ≤ 3.28; `train_time` lower is better.

| metric | value |
|---|---|
| val_loss | 3.2783 |
| train_time | 301,825 ms (≈ 5.03 min) |
| steps | 1875 |
| step_avg | 161.84 ms |

Windowed, document-masked FlexAttention at 64K context collapses the step count to 1875 (from 3000) at
roughly unchanged per-step time, cutting the wallclock to ~5.03 min. The README flags the trade: run-to-run
variance is now ~0.005 std and the mean (~3.279) clears the bar even though not every single run lands below
3.28. (Record #12, "1024-ctx dense causal attention → 64K-ctx FlexAttention".)
