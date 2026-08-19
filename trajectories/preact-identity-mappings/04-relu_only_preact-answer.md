**Problem.** ReLU-before-addition made `f` identity but forced `F >= 0`, losing to baseline —
the capacity restriction dominated. The two properties needed (`f` identity, `F` unrestricted
in sign) are not automatically compatible with a naive single-ReLU relocation: the previous
move bought the first by sacrificing the second.

**Key idea.** Break the symmetry differently. In the original design, `f(y_l)` feeds *both* the
next unit's shortcut and the next unit's branch identically: `y_{l+1} = f(y_l) + F(f(y_l),
W_{l+1})`. Define an asymmetric activation `f_hat` that touches only the branch path:
`y_{l+1} = y_l + F(f_hat(y_l), W_{l+1})`. The after-addition op between units is now a bare add
(`f` = identity), and `f_hat` has relocated to sit in front of the *next* unit's first weight
layer rather than after the current unit's addition — "asymmetric after-addition activation" and
"pre-activate the weight layers" are the same operation seen two ways. Since the branch still
*ends* at a weight layer's raw output, `F` regains its unrestricted range.

**Step-1 edit (smallest step in this direction).** Move only the ReLU to the front of each conv
in the branch; leave BN exactly where it already sits (after each conv), unchanged. Branch
becomes `ReLU -> conv -> BN -> ReLU -> conv -> BN`, nothing after the addition. This fixes both
diagnosed problems (shortcut-compounding from a merge-point BN; capacity restriction from a
branch-ending ReLU) without reintroducing either.

**Prediction.** At least matching baseline, since both previously-diagnosed failure modes are
removed. But the leading ReLU in this variant operates on the raw, unnormalized `x_l` — no BN
sits in front of it, so it doesn't get BN's normalization benefit the way every other weight
layer in the network does. Expect roughly baseline, not a clear win: three outcomes are
distinguishable and each says something different — matching baseline points at normalization
placement as the remaining gap; clearly worse means a third mechanism is still missing; clearly
better means the highway/range fixes alone were sufficient.

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
