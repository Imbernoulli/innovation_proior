Measured results — ReLU before addition, CIFAR-10 test set, median of 5 runs. `f` = identity
(bare add); ReLU relocated to the branch's last operation (`F = ReLU(...)`, forced non-negative).

## CIFAR-10

| network | baseline (f = ReLU) | ReLU before addition (f = identity, F >= 0) |
|---|---|---|
| ResNet-110 | 6.61 | 7.84 |
| ResNet-164 | 5.93 | 6.14 |

Worse than baseline on both architectures, despite `f` now being exactly identity. Diagnostic:
the residual branch's output is non-negative everywhere (min observed F output = 0.0), and the
forward-propagated feature is monotonically non-decreasing coordinate-wise with depth along
same-shape stretches — consistent with the capacity restriction from forcing `F(x) >= 0`.
