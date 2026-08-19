Measured results — dropout added to WRN-28-10 (cross-validated probability, ZCA preprocessing, same
protocol as the width-depth grid), compared against the grid's no-dropout WRN-28-10 result and against
the strongest thin reference.

| Method | CIFAR-10 | CIFAR-100 |
|---|---|---|
| pre-act-ResNet-164  | 5.46 | 24.33 |
| pre-act-ResNet-1001 | 4.92 | 22.71 |
| WRN-28-10           | 4.17 | 20.5 |
| WRN-28-10-dropout    | 4.39 | 20.0 |

CIFAR-10 error is worse with dropout than without (4.17 -> 4.39); CIFAR-100 error is better with
dropout than without (20.5 -> 20.0).
