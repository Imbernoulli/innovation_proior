**Question.** At fixed, comparable parameter budget and widening factor `k=2`, does the internal
kernel-size pattern of a residual block's convolutions matter, or is the incumbent two-`3x3` basic
block already close to as good as it gets?

**Design.** Six block variants, all pre-activation (`BN -> ReLU -> conv` before every convolution),
all with plane count held constant across the block (no bottleneck reduction/expansion), differing
only in the list `M` of kernel sizes: `B(3,3)` (incumbent), `B(3,1,3)`, `B(1,3,1)`, `B(1,3)`, `B(3,1)`,
`B(3,1,1)`. `1x1` convolutions cost roughly `1/9` the params/FLOPs of a `3x3` at the same plane count,
so the four variants with only one `3x3` (`B(1,3,1)`, `B(3,1)`, `B(1,3)`, `B(3,1,1)`) are trained at
depth 40 (a shared, deeper network, since each block is individually cheaper); the two variants that
keep two `3x3`s are trained shallower to land in the same parameter band — `B(3,3)` at depth 28,
`B(3,1,3)` at depth 22 (already deepened by its extra `1x1`, so it needs the least additional depth).
All six use `k=2`. CIFAR-10, median test error over 5 runs; wall-clock time per training epoch
recorded alongside.

Code — a block builder parameterized by the kernel-size list `M`, so all six variants are one class:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlockM(nn.Module):
    """Pre-activation residual block B(M): M is a list of kernel sizes, e.g. (3, 3),
    (3, 1, 3), (1, 3, 1), (1, 3), (3, 1), (3, 1, 1). Plane count is held constant
    across the block (no bottleneck reduction)."""

    def __init__(self, in_planes, out_planes, kernel_sizes, stride=1):
        super().__init__()
        self.equal_in_out = in_planes == out_planes
        planes = [in_planes] + [out_planes] * len(kernel_sizes)

        layers = []
        for i, ksz in enumerate(kernel_sizes):
            layers.append(nn.BatchNorm2d(planes[i]))
            layers.append(nn.ReLU(inplace=True))
            pad = ksz // 2
            s = stride if i == 0 else 1
            layers.append(nn.Conv2d(
                planes[i], planes[i + 1], kernel_size=ksz, stride=s,
                padding=pad, bias=False
            ))
        self.branch = nn.Sequential(*layers)

        self.shortcut = None
        if not self.equal_in_out:
            self.shortcut = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride,
                padding=0, bias=False
            )

    def forward(self, x):
        residual = self.branch(x)
        shortcut = x if self.equal_in_out else self.shortcut(x)
        return shortcut + residual


def make_block_type_net(block_type, num_classes=10):
    """block_type in {"B331","B313","B131","B13","B31","B311"} -> (kernel_sizes, depth)."""
    configs = {
        "B331": ((3, 3), 28),
        "B313": ((3, 1, 3), 22),
        "B131": ((1, 3, 1), 40),
        "B13":  ((1, 3), 40),
        "B31":  ((3, 1), 40),
        "B311": ((3, 1, 1), 40),
    }
    kernel_sizes, depth = configs[block_type]
    return WideNetM(depth, widen_factor=2, kernel_sizes=kernel_sizes, num_classes=num_classes)


class WideNetM(nn.Module):
    def __init__(self, depth, widen_factor, kernel_sizes, num_classes=10):
        super().__init__()
        # depth counts total conv layers through the 3 groups; blocks_per_group derived
        # from the block's own convolution count len(kernel_sizes).
        n_convs_per_block = len(kernel_sizes)
        blocks_per_group = (depth - 4) // (3 * n_convs_per_block)
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False)
        self.group1 = self._make_group(widths[0], widths[1], blocks_per_group, 1, kernel_sizes)
        self.group2 = self._make_group(widths[1], widths[2], blocks_per_group, 2, kernel_sizes)
        self.group3 = self._make_group(widths[2], widths[3], blocks_per_group, 2, kernel_sizes)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)

    def _make_group(self, in_planes, out_planes, count, stride, kernel_sizes):
        layers = [ResidualBlockM(in_planes, out_planes, kernel_sizes, stride)]
        for _ in range(1, count):
            layers.append(ResidualBlockM(out_planes, out_planes, kernel_sizes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.avg_pool2d(x, 8, 1, 0).flatten(1)
        return self.fc(x)
```

**Protocol.** CIFAR-10, horizontal flip + 4px reflected-pad random crop, ZCA whitening (the default
preprocessing for this exploratory phase). SGD Nesterov, lr 0.1, momentum 0.9, weight decay 5e-4,
batch 128, 200 epochs, lr x0.2 at epochs 60/120/160. Median test error over 5 runs, per-epoch
wall-clock recorded, for each of the six `block_type` configurations above. Decision rule: adopt
whichever block minimizes median CIFAR-10 test error; if results cluster tightly, prefer the
cheapest/fastest block among the near-ties rather than the nominal best.
