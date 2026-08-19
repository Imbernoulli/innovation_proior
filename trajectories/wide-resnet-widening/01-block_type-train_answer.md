I want to start from the piece of the scaffold I have been handed rather than from any grand new
idea: the residual block itself, `BN -> ReLU -> conv3x3 -> BN -> ReLU -> conv3x3`, added to an identity
or projection shortcut. That block is inherited from the pre-activation baseline, and before I let
myself spend a training budget deciding how much depth versus how much channel width a network should
carry, I want to check something narrower and prior to both: is the block's own internal shape even
well-chosen, at the level of "which convolutions, of what kernel size, in what order"? If I skip this
and just assume the two-`3x3` block is correct, everything downstream inherits that assumption
unexamined.

There are exactly three levers that increase a block's representational power — more convolutions,
wider convolutions, larger kernels — and I can retire the third immediately on evidence that predates
this problem entirely: VGG and Inception already showed that stacking small `3x3` filters beats
single larger ones at matched receptive field, so I fix every kernel in this design at `3x3` or `1x1`
and never revisit spatial extent again. That leaves convolution count and channel width, and I refuse
to test them in the same experiment, because a block that changes both at once can't tell me which
change did the work. So this first move isolates convolution *count and pattern* at a fixed, modest
width (`k=2`, enough to be past the degenerate thin regime without yet committing to how wide the
final network should be) and asks only: given a comparable total budget, does the specific mix of
`3x3` and `1x1` layers inside a block matter?

A `1x1` convolution at matched plane count costs roughly a ninth of a `3x3`'s parameters and FLOPs —
there's no bottleneck-style plane reduction here, channel count stays constant across the block, so
substituting a `1x1` for a `3x3` genuinely frees budget rather than just changing where a reduction
happens. That framing gives me a natural, non-arbitrary set of candidates to test alongside the
incumbent `B(3,3)`: `B(3,1,3)` sandwiches a cheap `1x1` between two `3x3`s; `B(1,3,1)` is the mirror,
expand-contract around a single `3x3`, structurally the "straightened" bottleneck shape without the
plane-count squeeze a real bottleneck uses; `B(1,3)` and `B(3,1)` alternate one `1x1` and one `3x3`,
differing only in order; `B(3,1,1)` is one `3x3` followed by two `1x1`s, in the spirit of
Network-in-Network bolted onto a residual unit. Six blocks total, none with zero `3x3` convolutions
(a block that never grows its receptive field per unit isn't answering the question I'm asking).

Fairness here means holding parameter count roughly comparable, not holding depth fixed — the four
one-`3x3` variants are individually cheaper than the two-`3x3` ones, so I let depth compensate: all
four one-`3x3` blocks (`B(1,3,1)`, `B(3,1)`, `B(1,3)`, `B(3,1,1)`) go into a shared depth-40 network,
while the two-`3x3` blocks go shallower to land in the same parameter neighborhood — `B(3,3)` at depth
28, `B(3,1,3)` at depth 22 since its extra `1x1` already deepens it somewhat. Everything trained at
`k=2`, CIFAR-10, five runs each and I report the median rather than a single run, because six
candidates compared on one seed apiece would let noise masquerade as a structural ranking. I also
record wall-clock time per epoch for every configuration: even at matched parameter count a block
built from many small sequential ops is not the same computational shape on a GPU as one built from
fewer, larger ops, and if width-versus-depth timing arguments matter later I want that measurement
starting from the very first experiment rather than asserted once and never checked again.

I don't expect a dramatic winner. The incumbent block already encodes what the field converged on —
pre-activation order, no bottleneck, small filters — so my honest prior is that most of these six
variants land close together, and if a `1x1`-substituted variant edges ahead it's more likely a
parameter-efficiency story than a real representational one. But that's a prediction, not a result I
get to assume, and it needs a controlled measurement before I build everything after it on top of
`B(3,3)`. The block builder below makes all six variants one parameterized class differing only in
`kernel_sizes`, so the comparison is mechanically apples-to-apples — same shortcut logic, same
pre-activation order, same training protocol, only the branch's kernel-size list and the depth needed
to match parameter count change between runs.

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


BLOCK_TYPE_CONFIGS = {
    "B331": ((3, 3), 28),
    "B313": ((3, 1, 3), 22),
    "B131": ((1, 3, 1), 40),
    "B13":  ((1, 3), 40),
    "B31":  ((3, 1), 40),
    "B311": ((3, 1, 1), 40),
}
```

I train each of the six at `k=2` on CIFAR-10 (ZCA-whitened, flip + 4px reflected-pad crop), SGD
Nesterov lr 0.1, weight decay 5e-4, batch 128, 200 epochs, lr x0.2 at 60/120/160, five runs, and take
the median test error, alongside per-epoch wall-clock. Whichever block minimizes median error becomes
the fixed block for every rung after this one — and if several are within noise of each other, I fall
back to the cheapest of the near-ties rather than chasing a difference that may not be real.
