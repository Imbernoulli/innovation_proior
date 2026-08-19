The grid resolves into a clear, if slightly more textured than expected, picture. At every depth I
tested, widening keeps paying off across the whole tested range — the 40-layer row alone runs
6.85 -> 5.33 -> 4.97 -> 4.66 on CIFAR-10 as `k` goes 1, 2, 4, 8, with no sign of flattening. Depth is
the more interesting axis: at fixed large `k` it keeps helping from 16 up through 28 layers — 4.81 at
16-8, down to 4.38 at 22-8, down to 4.17 at 28-10 — but pushing that same wide regime to 40 layers
reverses it, 40-8 landing at 4.66, worse than 22-8 despite twice the parameters. So depth's payoff in
the wide regime isn't unlimited either; it just tops out at a far shallower depth than the thin
regime's degradation point. The best cell overall is 28-10: 4.17% CIFAR-10, 20.50% CIFAR-100, 36.5M
parameters. I was hoping this cell would land close to the thin 1001-layer reference's 10.2M
parameters for a clean depth-versus-width comparison at matched budget — it doesn't; 36.5M is over
three times that. 40-4 at 8.9M is the much closer match on raw parameter count, and that's a separate
comparison for later. What 28-10 is, right now, is simply the best test error the grid found, and it's
sitting on two orders of magnitude more parameters than the thin end of this same family (0.6M at
40-1), trained so far with nothing beyond flip-and-crop augmentation and whatever regularization batch
normalization provides on its own.

I don't want to assume that's enough regularization for a network this large, because I already have a
directly measured reason not to. On an earlier CIFAR-10 network built specifically to test whether
batch normalization makes dropout redundant, stacking the two together reached 92.44% accuracy, and
pulling either one back out dropped it to 91.4%. That's a measured gap, not a worry stated in the
abstract, and it argues against the tidy claim that normalization alone closes the question — though
it was measured on a plain VGG-style network, not a residual one, so it doesn't transfer mechanically
here. It's enough to keep dropout on the table rather than assume it's superfluous, given that the
widening axis I just finished exploring multiplied parameter count by nearly two orders of magnitude
with zero added regularization to match.

Placement isn't a free choice, and I don't have to guess at the failure mode — there's a directly
relevant, already-measured result on a residual architecture of comparable depth to the deep end of
this family. Dropout at ratio 0.5 applied to the *output of the identity shortcut* failed to converge
to a good solution at all: past 20% test error against a 6.61% baseline on the same network. The stated
mechanism generalizes past that one ratio and past dropout specifically — any multiplicative
manipulation of the shortcut, scaling, gating, a 1x1 convolution standing in for identity, dropout,
imposes some expected damping on the one path meant to carry signal through the whole stack unimpeded,
and everything downstream of that damping inherits it. That rules the shortcut out categorically as a
location for regularization here, independent of the exact ratio or of this being a different network
than the one the result was measured on — the argument is about what a shortcut structurally is, not
about the specific architecture it was tested on. So dropout goes inside the residual branch, and given
the block's own `BN -> ReLU -> conv -> BN -> ReLU -> conv` structure, the natural slot is right after
the second `BN -> ReLU` and before the second convolution: it perturbs the residual transform on its
way through without ever touching the identity path, and it hands the next block's batch normalization
a less stale activation distribution than an unperturbed branch would.

I'm not going to import the 0.5 ratio from the shortcut-failure result — that number was measured for
a placement I've just ruled out, and there's no reason a ratio calibrated for a catastrophic-failure
mode should be right for a placement expected to behave well. So the probability itself gets
cross-validated on a held-out slice of training data rather than guessed, and whatever value that
selects gets applied to exactly the one configuration this rung is actually about — the grid's
best-found cell, 28-10 — rather than re-sweeping the whole grid with dropout added everywhere, which
would answer a much bigger question than the one currently in front of me.

I genuinely don't know which way this comes out, and I want to be honest about that rather than write
a prediction dressed up as a conclusion. The self-account argues dropout should help once there's
enough capacity to overfit into, and 36.5M parameters against 50,000 CIFAR-10 training images is real
capacity. But that result came from a different architecture with a different normalization
interaction, batch normalization is already doing some regularizing work here on its own, and dropout
sitting inside a residual branch — where the branch's output gets added back to an untouched shortcut,
rather than being the network's only forward path — is a genuinely different setting than either prior
result I'm drawing on. What I'm confident of is the placement: branch, not shortcut, so this shouldn't
fail the way the identity-mapping shortcut experiment failed. Whether it helps, by how much, and
whether that magnitude differs between CIFAR-10 and CIFAR-100 — that's what the measurement is for.

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

I cross-validate the dropout probability, train the resulting `wrn_28_10_with_dropout` under the same
ZCA-preprocessed CIFAR-10/CIFAR-100 protocol as the grid, and compare it directly against the
no-dropout 28-10 cell that protocol already measured.
