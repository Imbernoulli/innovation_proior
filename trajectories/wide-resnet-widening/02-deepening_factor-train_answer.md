The block-type sweep came back close to my prior: `B(3,3)` on top at 5.73%, with `B(3,1)` and
`B(3,1,3)` right behind at 5.78% each and cheaper. It was close enough that my own stated decision
rule — when results cluster, take the cheapest near-tie — would technically point at one of the
lighter variants. I'm overruling that on a reason outside the table itself: I'm about to spend most of
this design process scaling parameters by factors of ten to sixty on the width axis, and I want every
one of those configurations directly comparable to the two-`3x3`-convolution convention that every
published thin-residual number I have access to is built on. Saving 7-8% of parameters now is not
worth losing that comparability later, so `B(3,3)` is fixed, and I stop touching the internal kernel
pattern.

That leaves the other half of the original question completely open, and it's a genuinely different
question from the one I just closed. Block-type compared different kernel-size *patterns* at roughly
matched budgets chosen ad hoc per variant. It never isolated convolution *count* per block as its own
controlled axis. I want to do that cleanly now: with the block's internal kernels fixed at `3x3`,
what happens if I deepen each residual unit — stack more sequential convolutions inside one shortcut —
while shrinking the number of units so total convolution count and total parameter count stay exactly
fixed? Call the count `l`; the current default is `l=2`, and I'm going to measure it as a real point in
a sweep rather than assume it's already right.

There's a genuine tension I don't get to resolve by inspection. One story says larger `l` should win:
a deeper block can express a strictly richer per-unit transformation than a shallow one at the same
parameter count spent inside that unit, the same way a deeper plain network out-expresses a shallower
one at matched width — and since total depth is held fixed either way, why not always prefer fewer,
more-expressive units? The other story is the one this entire design starts from: the identity
shortcut. Every block is also a shortcut, an unimpeded path for both the forward activation and the
backward gradient, and stochastic depth's result — competitive accuracy even with whole blocks
randomly dropped during training — is direct, already-measured evidence that shortcuts are doing real
optimization work independent of what their residual branch computes. Raising `l` from 2 toward 3 or 4
does not add capacity for free at fixed total budget: every convolution moved inside a block is one
not spent instantiating another block, so the block count `d` must fall to compensate, and falling `d`
means fewer shortcuts. So the real question isn't "do deeper blocks help" as an isolated fact — it's
whether the per-unit expressiveness gained by raising `l` outweighs the shortcut density lost by the
corresponding drop in `d`. The stochastic-depth evidence says shortcuts matter, but it doesn't say by
how much relative to a modest per-block capacity increase, and I'd rather measure that trade than
assume which side of it wins.

I'm including `l=1` in the sweep too, not just comparing `l=2` against larger alternatives, because
`l=1` sits at the opposite extreme from `l=4`: the most blocks (most shortcuts) of any configuration
here, and the least per-unit expressiveness of any of them. If the shortcut-density story is right,
`l=1` should anchor one end of a clean pattern; if a single convolution per unit is simply too weak to
represent anything useful regardless of how many units there are, `l=1` should fail for a different
reason, and running the full range from 1 to 4 rather than a narrower comparison is how I'll be able to
tell those two failure modes apart instead of just seeing "one number worse" and guessing why.

Isolating `l` cleanly means fixing everything else: `k=2`, `3x3` convolutions throughout, and one
shared total-convolution-count / total-parameter budget — `WRN-40-2`'s own budget, about `2.2M`
parameters under the current `l=2` convention — that every value of `l` has to hit by adjusting the
block count `d` accordingly. So `l=1`, `l=2`, `l=3`, `l=4` all land at the same total depth and roughly
the same parameter count, and only the grouping of convolutions into units differs between them. Same
CIFAR-10 protocol as the block-type rung: ZCA whitening, flip and 4px reflected-pad crop augmentation,
SGD Nesterov, median test error over 5 runs to keep a single unlucky seed from being read as a
structural result. A monotonic fall in error as `l` rises would mean capacity-per-unit dominates and I
should keep deepening blocks in later rungs; a minimum at `l=2` or a rise past it would mean shortcut
density dominates and the two-convolution block stays fixed while every further budget increase goes
to channel width instead. I need that answer before I can honestly argue for spending the *next*
budget increase on width rather than depth, which is the actual point of everything after this rung.

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

I train `WideNetL(l, widen_factor=2, total_conv_layers=40)` for `l in {1, 2, 3, 4}` under the same
CIFAR-10 protocol as before and read off the median test error across `l` to settle which structural
story — richer units or denser shortcuts — actually governs this architecture at fixed budget.
