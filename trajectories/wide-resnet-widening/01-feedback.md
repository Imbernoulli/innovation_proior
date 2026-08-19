Measured results — block-type sweep, `k=2`, CIFAR-10, ZCA preprocessing, median test error over 5
runs; time column is wall-clock per training epoch.

| block type | depth | # params | time, s | CIFAR-10 test error, % |
|---|---|---|---|---|
| B(1,3,1) | 40 | 1.4M | 85.8 | 6.06 |
| B(3,1)   | 40 | 1.2M | 67.5 | 5.78 |
| B(1,3)   | 40 | 1.3M | 72.2 | 6.42 |
| B(3,1,1) | 40 | 1.3M | 82.2 | 5.86 |
| B(3,3)   | 28 | 1.5M | 67.5 | 5.73 |
| B(3,1,3) | 22 | 1.1M | 59.9 | 5.78 |
