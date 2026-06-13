Measured results — `baseline:bernnet` (`is_final,true`), seeds {42, 123, 456} and mean. Metric: mean
test accuracy over 10 splits (higher is better); the across-run std is shown alongside.

## Test accuracy (mean over 10 splits)
| seed | cora | citeseer | texas | cornell |
|---|---|---|---|---|
| 42 | 0.8563 | 0.7798 | 0.9049 | 0.8836 |
| 123 | 0.8552 | 0.7793 | 0.9082 | 0.8262 |
| 456 | 0.8547 | 0.7795 | 0.9148 | 0.8230 |
| **mean** | **0.8554** | **0.7795** | **0.9093** | **0.8443** |

## Across-run std (std of accuracy over the 10 splits)
| seed | cora | citeseer | texas | cornell |
|---|---|---|---|---|
| 42 | 0.0143 | 0.0061 | 0.0252 | 0.0377 |
| 123 | 0.0146 | 0.0058 | 0.0245 | 0.0403 |
| 456 | 0.0151 | 0.0048 | 0.0262 | 0.0445 |
| **mean** | **0.0147** | **0.0056** | **0.0253** | **0.0408** |
