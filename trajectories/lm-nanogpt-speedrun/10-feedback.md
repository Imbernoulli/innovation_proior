Measured result — record `2025-01-13_Fp8LmHead`, log `c51969c2...txt`. Final line of the reproducible run
log. `val_loss` lower is better, ≤ 3.28; `train_time` lower is better.

| metric | value |
|---|---|
| val_loss | 3.2770 |
| train_time | 188,512 ms (≈ 3.142 min) |
| steps | 1395 |
| step_avg | 136.11 ms |

Running the head matmul in FP8 dropped the per-step time from ≈148 ms to ≈136 ms while the step count
stayed essentially flat (1390 → 1395), confirming this was a per-step-time win on the biggest matmul rather
than a step-count change — wallclock fell to ≈3.142 min. (World-record table record #19, "FP8 head, offset
logits, lr decay to 0.1 instead of 0.0".)
