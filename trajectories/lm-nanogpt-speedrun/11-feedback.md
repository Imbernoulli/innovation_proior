Measured result — record `2025-08-23_SparseAttnGate`, log `020630eb...txt`. Final line of the reproducible
run log. `val_loss` lower is better, ≤ 3.28; `train_time` lower is better.

| metric | value |
|---|---|
| val_loss | 3.2772 |
| train_time | 168,938 ms (≈ 2.812 min) |
| steps | 1695 |
| step_avg | 99.67 ms |

The sparse per-head gate gives each head a distance-independent context-based no-op, removing the reliance
on the BoS attention sink; it adds a slight per-step cost but saves roughly 50 steps, and wallclock falls to
≈2.812 min. (The large per-step-time drop from rung 10's ≈136 ms to ≈99 ms also reflects intervening systems
work bundled into this era; the distinct method here is the sparse gate.) The record's README reports a
14-run validation: mean val loss 3.2787, std 0.0016, a one-sided t-test vs 3.28 giving p=0.0059 —
statistically below the bar. (World-record table record #28, "Sparse attention gate".)
