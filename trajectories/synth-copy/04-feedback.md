Measured results — `baseline:transformer_alibi` (`is_final,true`), seed 42. Per-variant columns; `score = 0.5·exact_match_id + 0.5·exact_match_ood`.

| variant | exact_match_id | token_acc_id | exact_match_ood | token_acc_ood | score | elapsed |
|---|---|---|---|---|---|---|
| delim | 1.0 | 1.0 | 0.570312 | 0.910123 | 0.785156 | 132.5 |
| repeat | 0.967773 | 0.97559 | 0.808594 | 0.886235 | 0.888184 | 135.5 |
| reverse | 0.994141 | 0.997573 | 0.25 | 0.268569 | 0.62207 | 129.5 |

Task-level summary (geometric mean of per-variant `score`): **0.7570**.
