Measured results — depth-vs-width grid, ZCA preprocessing, no dropout, CIFAR-10/CIFAR-100 test error.

| depth | k | # params | CIFAR-10 | CIFAR-100 |
|---|---|---|---|---|
| 40 | 1  | 0.6M  | 6.85 | 30.89 |
| 40 | 2  | 2.2M  | 5.33 | 26.04 |
| 40 | 4  | 8.9M  | 4.97 | 22.89 |
| 40 | 8  | 35.7M | 4.66 | - |
| 28 | 10 | 36.5M | 4.17 | 20.50 |
| 28 | 12 | 52.5M | 4.33 | 20.43 |
| 22 | 8  | 17.2M | 4.38 | 21.22 |
| 22 | 10 | 26.8M | 4.44 | 20.75 |
| 16 | 8  | 11.0M | 4.81 | 22.07 |
| 16 | 10 | 17.1M | 4.56 | 21.59 |

Note: the 40-8 cell was not run on CIFAR-100 (dash in the source table).
