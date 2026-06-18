Measured results — `baseline:lstm` (`is_final,true`), seed 42.

## dyck-k2-m3 (k=2, m=3)
| seed | id_token_acc | ood_token_acc | id_string_acc | ood_string_acc | params |
|---|---|---|---|---|---|
| 42 | 0.999863 | 0.999938 | 0.997 | 0.997 | 67334 |

## dyck-k8-m5 (k=8, m=5)
| seed | id_token_acc | ood_token_acc | id_string_acc | ood_string_acc | params |
|---|---|---|---|---|---|
| 42 | 0.932083 | 0.924061 | 0.126 | 0.001 | 68882 |

## dyck-length-ood (k=4, m=4, train ≤64, OOD 128–256)
| seed | id_token_acc | ood_token_acc | id_string_acc | ood_string_acc | params |
|---|---|---|---|---|---|
| 42 | 0.984675 | 0.973799 | 0.666 | 0.026 | 67850 |

Per-env score = `ood_token_acc`; task score = geometric mean across envs ≈ 0.962.
