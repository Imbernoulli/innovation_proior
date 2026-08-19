Measured results — full pre-activation, CIFAR-10 test set, median of 5 runs. `f` = identity
(bare add); both BN and ReLU moved to the front of each conv (`BN -> ReLU -> conv`, twice).

## CIFAR-10

| network | baseline (original unit) | ReLU-only pre-activation | full pre-activation |
|---|---|---|---|
| ResNet-110 | 6.61 | 6.71 | 6.37 |
| ResNet-164 | 5.93 | 5.91 | 5.46 |

First variant to clearly beat baseline on both architectures. Also cross-checked at a third,
smaller-unit configuration (110-layer, 1-layer skip per residual unit instead of 2): baseline
9.90, full pre-activation 8.91 — the improvement direction holds there too.
