Measured results — `baseline:lstm_attn` (`is_final,true`), seed 42. Per-variant columns; `score = 0.5·exact_match_id + 0.5·exact_match_ood`.

| variant | exact_match_id | token_acc_id | exact_match_ood | token_acc_ood | score | elapsed |
|---|---|---|---|---|---|---|
| delim | 1.0 | 1.0 | 0.0 | 0.371115 | 0.5 | 413.5 |
| repeat | 1.0 | 1.0 | 0.0 | 0.207253 | 0.5 | 523.2 |
| reverse | 1.0 | 1.0 | 0.0 | 0.179562 | 0.5 | 413.1 |

Task-level summary (geometric mean of per-variant `score`): **0.5000**.
