Measured results — `baseline:vgae` (`is_final,true`), seeds {42, 123, 456} and mean for Cora/CiteSeer;
ogbl-collab is seed 42 only.

## Cora
| seed | AUC | MRR | Hits@20 |
|---|---|---|---|
| 42 | 90.51 | 24.04 | 63.00 |
| 123 | 84.82 | 14.78 | 39.09 |
| 456 | 85.10 | 21.30 | 45.73 |
| **mean** | **86.81** | **20.04** | **49.27** |

## CiteSeer
| seed | AUC | MRR | Hits@20 |
|---|---|---|---|
| 42 | 91.46 | 46.24 | 73.19 |
| 123 | 84.78 | 19.42 | 45.27 |
| 456 | 84.35 | 15.78 | 41.98 |
| **mean** | **86.86** | **27.15** | **53.48** |

## ogbl-collab
| seed | Hits@50 | MRR |
|---|---|---|
| 42 | 31.77 | 7.70 |
