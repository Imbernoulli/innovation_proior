Measured results — 1001-layer network (333 pre-activated bottleneck units, 111 per feature-map
size), original unit vs. full pre-activation unit, median of 5 runs (mean +/- std in brackets
where reported).

## CIFAR-10

| network | original unit | pre-activation unit |
|---|---|---|
| ResNet-110 (1-layer skip) | 9.90 | 8.91 |
| ResNet-110 | 6.61 | 6.37 |
| ResNet-164 | 5.93 | 5.46 |
| ResNet-1001 | 7.61 | 4.92 (4.89 +/- 0.14) |

## CIFAR-100

| network | original unit | pre-activation unit |
|---|---|---|
| ResNet-164 | 25.16 | 24.33 |
| ResNet-1001 | 27.82 | 22.71 (22.68 +/- 0.22) |

The margin widens sharply with depth: +0.24 at ResNet-110, +0.47 at ResNet-164, +2.69 at
ResNet-1001 (CIFAR-10); +0.83 at ResNet-164 vs. +5.11 at ResNet-1001 (CIFAR-100). At 1001
layers, the original-unit network is now the worst of the four CIFAR-10 configurations shown
here (7.61%, worse than its own 110-layer counterpart's 6.61%) while the pre-activation network
is the best (4.92%). Training curves show the pre-activation network's training loss reduced
very quickly from the start of training at this depth, and reaching the lowest final training
loss among all configurations investigated; the original-unit network's training loss is reduced
only slowly at the start. Parameter count: ResNet-1001 has 10.2M parameters (both unit types,
same architecture skeleton). Training time: ResNet-1001 takes about 27 hours on 2 GPUs.
