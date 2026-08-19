The deepening-factor sweep resolved cleanly, and not in the direction pure capacity-per-unit would
predict: `l=2` won at 5.43%, `l=1` was too weak at 6.69%, and `l=3`/`l=4` got progressively worse
again, 5.65% and 5.93%. That's a peak, not a monotone climb — if a richer per-unit transform were the
dominant effect, `l=4` should have been best, and instead it's nearly the worst point in the sweep.
That shape matches the shortcut-density story: past `l=2`, trading blocks (shortcuts) for extra
per-unit depth costs more than it buys, at this fixed total budget, while `l=1` fails for the opposite
reason — one convolution just isn't enough transformation regardless of how many units carry it. Both
axes of the block's own internal shape are now settled: the kernel pattern barely matters (rung 1),
and `l=2` is a real optimum, not an unexamined default (rung 2). Everything left to spend goes into
either more blocks or wider ones — the axis this whole process has been aimed at from the start.

Depth's price is already on the table from prior art: the strongest thin reference spans 110 to 1001
layers at 1.7M to 10.2M parameters, and each further slice of accuracy in that family costs very
roughly doubling the layer count. Width's price I have to work out rather than assume, because the
direction isn't obvious without doing the arithmetic. Block count `d` enters parameter count linearly
— twice as many blocks, twice the parameters, channels held fixed. But the widening factor `k`
multiplies every convolution's channel count on both sides, and a convolution's parameter count runs
roughly as `in_channels x out_channels x kernel_area`: doubling `k` doubles both channel dimensions of
every convolution in every block, so the cost is closer to `4x` per doubling, not `2x`. Concretely:
the shallowest skeleton in this family, 16 layers, at `k=1` gives stage widths `16, 16, 32, 64` —
essentially the thin baseline, well under a quarter-million parameters. The same 16-layer skeleton at
`k=8` gives widths `16, 128, 256, 512` and lands around `11M` parameters — roughly sixty times more for
eight times the width. Eight-fold widening, sixty-fold parameters: the quadratic scaling shows up
exactly where the algebra says it should, because `k` enters both ends of nearly every convolution.
Width is genuinely expensive per unit of widening, on paper. So the case for spending budget on it
can't be a parameter-efficiency argument — depth already wins that one by construction. It has to be
that width buys something per parameter, or per unit of training time, that depth can't.

That something is wall-clock, and I don't have to assert it — rung 1 already measured a hint of it,
even though timing wasn't what that rung was testing. At comparable parameter count, the two-`3x3`,
fewer-and-bigger-block variants trained about as fast or faster than the deeper, cheaper-per-block
ones: `B(3,3)` at depth 28 ran 67.5s/epoch, matching `B(3,1)` at depth 40 despite `B(3,1)` stepping
through more sequential blocks, and `B(3,1,3)` at the shallowest depth in that comparison, 22 layers,
was the fastest of the six at 59.9s/epoch. That's consistent with the general GPU argument: many small
sequential operations chained end to end underuse hardware built for large parallel workloads, while
fewer, larger convolutions give it more to do per launch. If that pattern holds at the much larger
scale I'm about to test, the comparison that actually matters isn't accuracy-per-parameter, where
depth already wins by construction — it's accuracy-per-training-time, where a quadratically more
expensive but more parallel width axis might still come out ahead.

I don't know the shape of that tradeoff yet at the scale that matters, so I want a grid, not one
comparison. Two questions are tangled together in "should I go wide," and they need to be separated:
at a fixed, modest depth, does raising `k` keep helping, or does it saturate or start to overfit,
given only flip-and-crop augmentation and no regularization added yet? And at a fixed, already-wide
`k`, does depth still help the way it does in the thin regime, or does the degradation that extreme
thin depth shows on its own reappear once the network is already wide? A single point can't distinguish
a `k`-effect from a depth-effect from their interaction — only a grid over both axes can. So: depths
from 16 up to 40 (the working budget the last two rungs established), and `k` from 2 up to 12, wide
enough that if returns diminish somewhere in this range I'd actually see it rather than stopping short.
Every cell reports both CIFAR-10 and CIFAR-100, since CIFAR-100 — ten times fewer images per class — is
the sharper overfitting stress test at this parameter scale; if unregularized width is going to show
diminishing or negative returns anywhere in this budget, CIFAR-100 is where I'd expect to see it first.

One comparison in particular is the sharpest test of the idea this whole process has been circling —
that the depth-to-width ratio thin residual networks settled on is one expensive point in the design
space, not a structural necessity. The strongest thin reference sits at 1001 layers and 10.2M
parameters. If some cell in this grid — thirty to forty layers, not a thousand — lands in a comparable
parameter neighborhood and gets a competitive or better error while training in a fraction of the
time the timing argument above predicts, that's real evidence against the idea that extreme depth
itself provides some regularization or representational benefit width can't substitute for. I'm not
assuming that outcome — the grid is what decides it — but it's the specific readout I'll be pulling out
once the numbers exist, alongside the more basic question of where in the `(depth, k)` plane CIFAR-10
and CIFAR-100 error actually bottom out within this range.

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

I sweep every `(depth, k)` cell in `GRID` on both CIFAR-10 and CIFAR-100, still ZCA-preprocessed, no
dropout, and read off two things once the numbers exist: whether error keeps falling as `k` rises at
fixed depth or saturates, and whether any 16-to-40-layer cell reaches the thin 1001-layer reference's
parameter neighborhood while matching or beating its accuracy — the comparison this whole design
process has been building toward.
