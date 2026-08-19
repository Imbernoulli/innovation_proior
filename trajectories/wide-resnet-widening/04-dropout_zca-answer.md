**Question.** Does adding dropout inside the residual branch of the grid's best-found configuration
(WRN-28-10, 36.5M params) help, hurt, or do nothing, under the same ZCA-preprocessed protocol the grid
was measured under?

**Design.** Dropout placed strictly inside the residual branch — after the second `BN -> ReLU`, before
the second convolution — never on the shortcut (ruled out categorically: any multiplicative
manipulation of the identity path, dropout included, is documented to hamper signal propagation).
Dropout probability chosen by cross-validation on a held-out slice of the CIFAR-10 training set (a
small grid of candidate probabilities, e.g. `{0.1, 0.2, 0.3, 0.4, 0.5}`, evaluated on the held-out
split; the identity-mapping shortcut-dropout ratio of 0.5 is not reused here since it was measured for
a different placement that is already known to fail). No change to training epoch budget or learning
rate schedule versus the no-dropout run.

Code — dropout added to the `B(3,3)` block at the one location the reasoning settles on:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBasicBlockDropout(nn.Module):
    """B(3,3) block with dropout inside the residual branch only (never on the shortcut):
    BN -> ReLU -> conv -> BN -> ReLU -> dropout -> conv."""

    def __init__(self, in_planes, out_planes, stride=1, dropout=0.0):
        super().__init__()
        self.equal_in_out = in_planes == out_planes
        self.dropout = dropout
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
        residual = F.relu(self.bn2(residual), inplace=True)
        if self.dropout > 0:
            residual = F.dropout(residual, p=self.dropout, training=self.training)
        residual = self.conv2(residual)
        shortcut = x if self.equal_in_out else self.shortcut(pre)
        return shortcut + residual


def wrn_28_10_with_dropout(dropout, num_classes=10):
    depth, widen_factor = 28, 10
    n = (depth - 4) // 6
    widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False)
            self.group1 = self._make_group(widths[0], widths[1], n, 1)
            self.group2 = self._make_group(widths[1], widths[2], n, 2)
            self.group3 = self._make_group(widths[2], widths[3], n, 2)
            self.bn = nn.BatchNorm2d(widths[3])
            self.fc = nn.Linear(widths[3], num_classes)

        def _make_group(self, in_planes, out_planes, count, stride):
            layers = [WideBasicBlockDropout(in_planes, out_planes, stride, dropout)]
            for _ in range(1, count):
                layers.append(WideBasicBlockDropout(out_planes, out_planes, 1, dropout))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.group1(x)
            x = self.group2(x)
            x = self.group3(x)
            x = F.relu(self.bn(x), inplace=True)
            x = F.avg_pool2d(x, 8, 1, 0).flatten(1)
            return self.fc(x)

    return Net()
```

**Protocol.** Cross-validate dropout probability on a held-out slice of CIFAR-10 training data; train
`wrn_28_10_with_dropout(p)` at the selected `p` with the same ZCA-preprocessed CIFAR-10/CIFAR-100
protocol, augmentation, and optimizer schedule as the grid (SGD Nesterov lr 0.1, weight decay 5e-4,
batch 128, 200 epochs, lr x0.2 at 60/120/160, median of 5 runs), no change to epoch budget. Compare
directly against the grid's no-dropout WRN-28-10 result on both CIFAR-10 and CIFAR-100.
