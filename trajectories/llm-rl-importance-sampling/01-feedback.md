Measured `baseline:first_k_tokens`, seed 42, from the latest complete matched-config baseline triple (2026-04-21T14:47, full 100-step pipeline, single H200, elapsed 21411 s). This task runs one seed; the baseline rows carry `is_final,false` by design (the baselines do not call `submit()`), so these are the real measured `baseline:*` numbers, not consensus estimates. Higher `score_mean` is better.

| metric | seed 42 |
|---|---|
| score_mean | −0.751 |
| gsm8k_accuracy | 0.4391 |
| math500_accuracy | 0.2875 |
| amc_accuracy | 0.0828 |

Single seed (42); the per-seed value is the mean.
