Measured results — `baseline:seal` (`is_final,true`), seeds {42, 123, 456} and mean for Cora/CiteSeer;
ogbl-collab is seed 42 only.

## Cora
| seed | AUC | MRR | Hits@20 |
|---|---|---|---|
| 42 | 92.16 | 23.46 | 60.72 |
| 123 | 92.32 | 21.49 | 54.84 |
| 456 | 93.02 | 37.52 | 67.93 |
| **mean** | **92.50** | **27.49** | **61.16** |

## CiteSeer
| seed | AUC | MRR | Hits@20 |
|---|---|---|---|
| 42 | 93.86 | 31.64 | 73.63 |
| 123 | 93.01 | 34.29 | 74.73 |
| 456 | 91.69 | 39.73 | 67.91 |
| **mean** | **92.85** | **35.22** | **72.09** |

## ogbl-collab
| seed | Hits@50 | MRR |
|---|---|---|
| 42 | 57.88 | 10.58 |
