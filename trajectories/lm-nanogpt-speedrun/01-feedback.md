Measured result — record `2024-10-10_Muon` (the Muon-on-the-body record). Final line of the reproducible
run log. `val_loss` lower is better and must be ≤ 3.28; `train_time` lower is better (this is the raced
quantity).

| metric | value |
|---|---|
| val_loss | 3.2785 |
| train_time | 1,339,067 ms (≈ 22.3 min) |
| steps | 6200 |
| step_avg | 216.33 ms |

Down from the ~31-minute AdamW + rotary baseline: the orthogonalized-momentum update reaches the same
3.28 bar in materially less wallclock. (Per the repository's world-record table this is record #4, "Muon
improvements", 22.3 min; the bare Muon introduction was record #3 at 24.9 min, which ships no log.)
