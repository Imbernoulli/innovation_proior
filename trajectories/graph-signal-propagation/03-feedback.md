Measured results — `baseline:gprgnn` (`is_final,true`), seeds {42, 123, 456} and mean. Metric: mean
test accuracy over 10 splits (higher is better); the across-run std is shown alongside. (The cora seed
123 run is absent in the leaderboard; its mean is over the two recorded seeds.)

## Test accuracy (mean over 10 splits)
| seed | cora | citeseer | texas | cornell |
|---|---|---|---|---|
| 42 | 0.8887 | 0.8020 | 0.9098 | 0.8738 |
| 123 | — | 0.8020 | 0.9049 | 0.8738 |
| 456 | 0.8893 | 0.8019 | 0.9049 | 0.8639 |
| **mean** | **0.8890** | **0.8020** | **0.9065** | **0.8705** |

## Across-run std (std of accuracy over the 10 splits)
| seed | cora | citeseer | texas | cornell |
|---|---|---|---|---|
| 42 | 0.0095 | 0.0136 | 0.0471 | 0.0303 |
| 123 | — | 0.0136 | 0.0350 | 0.0434 |
| 456 | 0.0111 | 0.0136 | 0.0262 | 0.0275 |
| **mean** | **0.0103** | **0.0136** | **0.0361** | **0.0337** |
