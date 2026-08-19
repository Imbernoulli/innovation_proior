**Question.** With the block fixed (`B(3,3)`, `l=2`), how should a fixed-ish training budget be spent
between depth `d` and widening factor `k`? Sweep both together rather than one at a time, since a
single point in the `(depth, k)` plane can't separate a depth-effect from a `k`-effect or their
interaction.

**Design.** A depth-by-width grid: depths `{16, 22, 28, 40}`, and at each depth a range of `k` chosen
so parameter counts across the grid land in a few broadly comparable bands rather than one depth
running away in parameter count while another stays small — `k in {1, 2, 4, 8}` at depth 40 (spanning
from the thin baseline up through a substantial widening at the deepest network in the grid), `k in
{8, 10, 12}` at depth 28, `k in {8, 10}` at depths 22 and 16 (the shallower networks only tested at
already-wide `k`, since the point of including shallow depths is to see whether width alone can
compensate for lost depth, not to re-confirm that a shallow-and-narrow network is weak, which rung 2
already established at `l=1`). CIFAR-10 and CIFAR-100 test error reported for every cell, ZCA
preprocessing (still the exploratory-phase default), no dropout at this rung.

Code — a generic `(depth, widen_factor)` constructor, the same `B(3,3)` block as before, now exposed
as the two free knobs of the whole grid:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBasicBlock(nn.Module):
    """Pre-activation B(3,3) block, l=2, settled by the block-type and deepening-factor rungs."""

    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.equal_in_out = in_planes == out_planes
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.shortcut = None
        if not self.equal_in_out:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                       padding=0, bias=False)

    def forward(self, x):
        pre = F.relu(self.bn1(x), inplace=True)
        residual = self.conv1(pre)
        residual = self.conv2(F.relu(self.bn2(residual), inplace=True))
        shortcut = x if self.equal_in_out else self.shortcut(pre)
        return shortcut + residual


class WideResNetGrid(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=10):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        widths = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False)
        self.group1 = self._make_group(widths[0], widths[1], n, 1)
        self.group2 = self._make_group(widths[1], widths[2], n, 2)
        self.group3 = self._make_group(widths[2], widths[3], n, 2)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)

    def _make_group(self, in_planes, out_planes, count, stride):
        layers = [WideBasicBlock(in_planes, out_planes, stride)]
        for _ in range(1, count):
            layers.append(WideBasicBlock(out_planes, out_planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.avg_pool2d(x, 8, 1, 0).flatten(1)
        return self.fc(x)


GRID = {
    40: [1, 2, 4, 8],
    28: [10, 12],
    22: [8, 10],
    16: [8, 10],
}
```

**Protocol.** For every `(depth, k)` pair in `GRID`, train `WideResNetGrid(depth, k)` on both CIFAR-10
and CIFAR-100 (ZCA whitening, flip + 4px reflected-pad crop; SGD Nesterov lr 0.1, weight decay 5e-4,
batch 128, 200 epochs, lr x0.2 at 60/120/160), no dropout. Report test error per cell. Two specific
readouts to extract once the grid is in: (1) at each fixed depth, is error monotonically decreasing in
`k` across the tested range, or does it saturate/reverse; (2) at each fixed large `k` (8 or 10, the
only values shared across every depth in the grid), how does error move as depth rises from 16 to 40 —
and whether any cell in this 16-40-layer range reaches a parameter count comparable to the 1001-layer,
10.2M-parameter thin reference while beating or matching its error.
