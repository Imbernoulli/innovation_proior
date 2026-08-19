ReLU-before-addition made `f` identity but lost to baseline on both architectures (7.84% vs.
6.61% on ResNet-110, 6.14% vs. 5.93% on ResNet-164), and the diagnostic confirmed exactly the
mechanism I flagged going in: the branch output is non-negative everywhere, and the feature is
monotonically non-decreasing with depth along same-shape stretches. That's a capacity
restriction, not an optimization-difficulty failure in the shortcut-sweep sense — forcing `F(x)
>= 0` cuts out half of what a residual function needs to express, and that restriction won out
over whatever the clean highway bought.

This tells me the two properties I want — `f` identity, and `F` free to be negative — are not
automatically compatible with a naive single-ReLU relocation. The naive move bought the first
property by sacrificing the second. I need a relocation that gets both, and I should work out
what that structurally requires rather than guess again.

Go back to the two-unit picture. In the original design, `f(y_l)` feeds both the next unit's
shortcut and the next unit's branch identically: `y_{l+1} = f(y_l) + F(f(y_l), W_{l+1})` — the
same activation, symmetric across both paths. That symmetry is what trapped me: making the
after-addition op identity meant finding somewhere else for the activation to live, and the
naive move put it entirely at the branch's exit, breaking the branch's range. But the symmetry
itself is what I should break. Define an activation `f_hat` that touches only the branch path:
`y_{l+1} = y_l + F(f_hat(y_l), W_{l+1})`. The after-addition operation is now a bare add — `f`
is genuinely identity — and `f_hat` hasn't disappeared, it's moved to sit in front of the *next*
unit's first weight layer. Renaming back to `x`: `x_{l+1} = x_l + F(f_hat(x_l), W_l)`, the same
additive form I've been chasing. Crucially this doesn't force anything about what the branch
*ends* with, so it can still be negative.

"Asymmetric after-addition activation" and "pre-activate the weight layers" are the same
operation described two ways — the second phrasing is constructive: it says where to put the
activation. It's meaningless to call an activation "pre" or "post" in a plain unbranched stack
(there's no fact of the matter distinguishing the two); it's specifically the branch-and-merge
structure, needing one of the two paths to stay clean, that makes the position matter at all.

That still leaves a real choice: each weight layer currently has both a BN and a ReLU
associated with it, and "pre-activate" alone doesn't say whether to move just the ReLU, just
the BN, or both. I want to walk this incrementally rather than jump to moving everything —
BN-after-addition and ReLU-before-addition already failed for two *different* reasons that I
only told apart by testing both, so I don't trust myself to reason straight to the right
combination without checking. Smallest step: move only the ReLU to the front of each conv,
leave BN exactly where it sits (after each conv). Branch becomes `ReLU -> conv -> BN -> ReLU ->
conv -> BN`, nothing after the addition. Both diagnosed problems are fixed — merge-point BN
gone, branch-ending ReLU gone — without reintroducing either.

I expect this to at least match baseline. But I want to flag a reason it might not clearly beat
it: the leading ReLU in this variant operates on raw, unnormalized `x_l` — there's no BN sitting
in front of it the way there is in front of every other weight layer in the network. So I've
fixed the two structural problems I already diagnosed, but I haven't necessarily captured
whatever benefit BN's normalization provides to a weight layer's input. My prediction is roughly
baseline: matching would point at normalization placement as the remaining gap and motivate
moving BN forward too; clearly worse would mean a third mechanism I haven't accounted for;
clearly better would mean the highway and range fixes alone were sufficient. I don't know which
yet — that's what running it tells me.

```python
import torch.nn as nn
import torch.nn.functional as F


class ReLUOnlyPreActBlock(nn.Module):
    """ReLU moved to the front of each weight layer (pre-activation); BN
    left after each conv, unchanged from the original unit. f is identity
    (bare add); F ends in a conv, so it is unrestricted in sign."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.bn1(self.conv1(F.relu(x)))       # ReLU before conv1 (no BN in front of it)
        out = self.bn2(self.conv2(F.relu(out)))      # ReLU before conv2, BN before that ReLU
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        return shortcut + out                         # bare add: f is identity
```
