Measured results — ReLU-only pre-activation, CIFAR-10 test set, median of 5 runs. `f` = identity
(bare add); ReLU moved to the front of each conv; BN left after each conv, unchanged.

## CIFAR-10

| network | baseline (original unit) | ReLU-only pre-activation |
|---|---|---|
| ResNet-110 | 6.61 | 6.71 |
| ResNet-164 | 5.93 | 5.91 |

Performs very similar to baseline on both architectures — essentially a wash (slightly worse on
ResNet-110, slightly better on ResNet-164, both within noise of the median-of-5 protocol). This
leading ReLU is not used in conjunction with a BN layer immediately in front of it.
