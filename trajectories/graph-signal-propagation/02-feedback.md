Measured results — `baseline:chebnetii` (`is_final,true`), seeds {42, 123, 456} and mean. Metric: mean
test accuracy over 10 splits (higher is better); the across-run std is shown alongside.

## Test accuracy (mean over 10 splits)
| seed | cora | citeseer | texas | cornell |
|---|---|---|---|---|
| 42 | 0.8724 | 0.8001 | 0.8770 | 0.8475 |
| 123 | 0.8726 | 0.8007 | 0.8770 | 0.8475 |
| 456 | 0.8724 | 0.8004 | 0.8770 | 0.8459 |
| **mean** | **0.8725** | **0.8004** | **0.8770** | **0.8470** |

## Across-run std (std of accuracy over the 10 splits)
| seed | cora | citeseer | texas | cornell |
|---|---|---|---|---|
| 42 | 0.0161 | 0.0153 | 0.0246 | 0.0464 |
| 123 | 0.0161 | 0.0147 | 0.0246 | 0.0464 |
| 456 | 0.0161 | 0.0148 | 0.0246 | 0.0459 |
| **mean** | **0.0161** | **0.0149** | **0.0246** | **0.0462** |
