Measured results — `baseline:transformer` (`is_final,true`), seed 42.

## dyck-k2-m3 (k=2, m=3)
| seed | id_token_acc | ood_token_acc | id_string_acc | ood_string_acc | params |
|---|---|---|---|---|---|
| 42 | 0.959956 | 0.903061 | 0.465 | 0.001 | 266246 |

## dyck-k8-m5 (k=8, m=5)
| seed | id_token_acc | ood_token_acc | id_string_acc | ood_string_acc | params |
|---|---|---|---|---|---|
| 42 | 0.804292 | 0.727853 | 0.024 | 0.0 | 267794 |

## dyck-length-ood (k=4, m=4, train ≤64, OOD 128–256)
| seed | id_token_acc | ood_token_acc | id_string_acc | ood_string_acc | params |
|---|---|---|---|---|---|
| 42 | 0.924174 | 0.734313 | 0.277 | 0.0 | 266762 |

Per-env score = `ood_token_acc`; task score = geometric mean across envs ≈ 0.785.
