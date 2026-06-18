Measured results — `baseline:transformer_nope` (`is_final,true`), seed 42. Per-variant columns; `score = 0.5·exact_match_id + 0.5·exact_match_ood`.

| variant | exact_match_id | token_acc_id | exact_match_ood | token_acc_ood | score | elapsed |
|---|---|---|---|---|---|---|
| delim | 0.998047 | 0.99972 | 0.296875 | 0.433462 | 0.647461 | 130.3 |
| repeat | 0.996094 | 0.995613 | 0.549805 | 0.757108 | 0.772949 | 133.8 |
| reverse | 1.0 | 1.0 | 0.40332 | 0.434075 | 0.70166 | 130.8 |

Task-level summary (geometric mean of per-variant `score`): **0.7055**.
