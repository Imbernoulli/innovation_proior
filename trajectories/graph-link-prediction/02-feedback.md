Measured results — `baseline:gcn_dot` (`is_final,true`), seeds {42, 123, 456} and mean for
Cora/CiteSeer; ogbl-collab is seed 42 only.

## Cora
| seed | AUC | MRR | Hits@20 |
|---|---|---|---|
| 42 | 93.11 | 39.45 | 76.66 |
| 123 | 90.47 | 16.10 | 68.50 |
| 456 | 90.31 | 38.00 | 65.65 |
| **mean** | **91.30** | **31.18** | **70.27** |

## CiteSeer
| seed | AUC | MRR | Hits@20 |
|---|---|---|---|
| 42 | 93.05 | 39.40 | 75.82 |
| 123 | 90.51 | 36.46 | 72.75 |
| 456 | 88.83 | 46.67 | 72.53 |
| **mean** | **90.80** | **40.84** | **73.70** |

## ogbl-collab
| seed | Hits@50 | MRR |
|---|---|---|
| 42 | 53.74 | 13.79 |
