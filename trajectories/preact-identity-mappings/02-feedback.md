Measured results — BN after addition, CIFAR-10 test set, median of 5 runs. `f` = BN then ReLU
(applied to the merged signal); branch and shortcut otherwise unchanged from the original unit.

## CIFAR-10

| network | baseline (f = ReLU) | BN after addition (f = BN, ReLU) |
|---|---|---|
| ResNet-110 | 6.61 | 8.17 |
| ResNet-164 | 5.93 | 6.50 |

Worse than baseline on both architectures. Training curves show the BN layer now alters the
signal that passes through what becomes the next unit's shortcut, with visibly higher training
loss than baseline in the early phase of training on ResNet-110.
