Measured results — `baseline:transformer_sinusoidal` (`is_final,true`), seed 42. Per-variant columns; `score = 0.5·exact_match_id + 0.5·exact_match_ood`.

| variant | exact_match_id | token_acc_id | exact_match_ood | token_acc_ood | score | elapsed |
|---|---|---|---|---|---|---|
| delim | 1.0 | 1.0 | 0.0 | 0.071051 | 0.5 | 127.7 |
| repeat | 1.0 | 1.0 | 0.0 | 0.066312 | 0.5 | 131.1 |
| reverse | 0.999023 | 0.999907 | 0.000977 | 0.031206 | 0.5 | 123.7 |

Task-level summary (geometric mean of per-variant `score`): **0.5000**.
