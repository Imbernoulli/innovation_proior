Measured results — `baseline:stack_rnn` (`is_final,true`), seed 42.

## dyck-k2-m3 (k=2, m=3)
| seed | id_token_acc | ood_token_acc | id_string_acc | ood_string_acc | params |
|---|---|---|---|---|---|
| 42 | 1.0 | 1.0 | 1.0 | 1.0 | 17865 |

## dyck-k8-m5 (k=8, m=5)
| seed | id_token_acc | ood_token_acc | id_string_acc | ood_string_acc | params |
|---|---|---|---|---|---|
| 42 | 1.0 | 1.0 | 1.0 | 1.0 | 20181 |

## dyck-length-ood (k=4, m=4, train ≤64, OOD 128–256)
| seed | id_token_acc | ood_token_acc | id_string_acc | ood_string_acc | params |
|---|---|---|---|---|---|
| 42 | 1.0 | 1.0 | 1.0 | 1.0 | 18637 |

Per-env score = `ood_token_acc`; task score = geometric mean across envs = 1.0.
