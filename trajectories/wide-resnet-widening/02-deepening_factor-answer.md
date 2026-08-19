**Question.** At fixed total convolution count and fixed total parameter budget (`WRN-40-2`'s own
budget, `k=2`, `3x3` convolutions, `~2.2M` params), does packing more convolutions into each residual
block (`l`) help, at the cost of fewer blocks/shortcuts (`d`), or does shortcut density dominate?

**Design.** Block fixed to `B(3,3)`-style, i.e. every internal convolution is `3x3` (settled by the
block-type sweep). Sweep `l in {1, 2, 3, 4}` — the number of convolutions inside one residual unit —
while holding total convolution count and total parameter count constant across the sweep by adjusting
the number of blocks per group `d` inversely with `l`. `l=2` is the incumbent default, now measured
as its own point in the sweep rather than assumed to be best. CIFAR-10, median test error over 5
runs, same training protocol as the block-type rung.

Code — the block generalizes to arbitrary `l` (an `l`-deep stack of `3x3` pre-activation convolutions
inside one shortcut), and the network solves for `d` (blocks per group) given a target total
convolution count so every `l` in the sweep lands at the same total depth:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LDeepResidualBlock(nn.Module):
    """Pre-activation residual block with l stacked 3x3 convolutions inside one
    shortcut (l=2 recovers the incumbent B(3,3) block)."""

    def __init__(self, in_planes, out_planes, l, stride=1):
        super().__init__()
        self.equal_in_out = in_planes == out_planes
        planes = [in_planes] + [out_planes] * l

        layers = []
        for i in range(l):
            layers.append(nn.BatchNorm2d(planes[i]))
            layers.append(nn.ReLU(inplace=True))
            s = stride if i == 0 else 1
            layers.append(nn.Conv2d(
                planes[i], planes[i + 1], kernel_size=3, stride=s,
                padding=1, bias=False
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


def blocks_per_group_for_budget(l, total_conv_layers, n_groups=3):
    """Solve for d (blocks per group) so l * d * n_groups + 1 (stem) ~= total_conv_layers,
    holding total convolution count fixed across the l-sweep."""
    d = (total_conv_layers - 1) // (l * n_groups)
    assert d >= 1, f"l={l} leaves no room for a block at this budget"
    return d


class WideNetL(nn.Module):
    def __init__(self, l, widen_factor, total_conv_layers, num_classes=10):
        super().__init__()
        d = blocks_per_group_for_budget(l, total_conv_layers)
        k = widen_factor
        widths = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False)
        self.group1 = self._make_group(widths[0], widths[1], d, 1, l)
        self.group2 = self._make_group(widths[1], widths[2], d, 2, l)
        self.group3 = self._make_group(widths[2], widths[3], d, 2, l)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)

    def _make_group(self, in_planes, out_planes, count, stride, l):
        layers = [LDeepResidualBlock(in_planes, out_planes, l, stride)]
        for _ in range(1, count):
            layers.append(LDeepResidualBlock(out_planes, out_planes, l, 1))
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

**Protocol.** For each `l in {1, 2, 3, 4}`, build `WideNetL(l, widen_factor=2, total_conv_layers=40)`
(the total-layer budget of `WRN-40-2`, `~2.2M` params under `l=2`) and train on CIFAR-10 (ZCA
whitening, flip + 4px reflected-pad crop; SGD Nesterov lr 0.1, weight decay 5e-4, batch 128, 200
epochs, lr x0.2 at 60/120/160), median test error over 5 runs. Decision rule: if error falls
monotonically as `l` rises, capacity-per-unit dominates and the block should be deepened in later
rungs; if error is minimized at `l=2` or rises past it, shortcut density dominates and `l=2` stays
fixed while the remaining budget goes to width in the next rung.
