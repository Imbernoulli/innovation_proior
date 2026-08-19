**Problem.** BN-after-addition confirmed the after-addition operation `f` reaches the same
shortcut path the sweep already condemned, generalizing the mechanism. The actual target is now
clear: make `f` identity, the way `h` already is, so the additive highway holds exactly rather
than approximately. But `f` = ReLU can't simply be deleted — the network needs its
nonlinearities.

**Key idea.** The most conservative relocation: move the single existing after-addition ReLU to
immediately *before* the addition instead of after it. `x_{l+1} = ReLU(F(x_l)) + x_l` instead of
`x_{l+1} = ReLU(F(x_l) + x_l)`. The addition is now bare, so `f` is exactly identity and the
clean highway holds. Nothing else about the unit changes.

**Risk flagged before testing.** Putting ReLU as the branch's last operation forces
`F(x_l) = ReLU(...) >= 0` — a hard constraint on the function class, not a training-difficulty
story. Along a same-shape stretch this makes `x_L = x_l + sum F` a sum of non-negative terms
(monotone non-decreasing with depth), and removes the branch's ability to *decrease* a feature
relative to its shortcut value — directly cutting against the residual framing (`F` fitting
`H(x)-x`, which has no reason to be one-signed).

**Step-1 edit.** Single relocation of the existing ReLU; branch's two BNs, the identity/1x1
shortcut, and everything outside this unit unchanged. Test on ResNet-110 and ResNet-164, same
as the previous rung.

**What this decides.** Whether the clean-highway effect alone (with the branch's negative range
sacrificed) is enough to beat baseline, before trying anything that protects both properties at
once. Two failure modes are distinguishable here: an optimization-difficulty failure (elevated
training error, the sweep's signature) vs. a capacity restriction (a converged model that simply
can't represent the needed one-sided-inverted mappings) — the two look different and the
distinction is itself informative regardless of which way the number lands.

```python
import torch.nn as nn
import torch.nn.functional as F


class ReLUBeforeAddBlock(nn.Module):
    """ReLU relocated to the end of the branch (before the add) instead of
    after it. f is now exactly identity; F is constrained to [0, inf)."""
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
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        out = F.relu(out)                          # ReLU is now the branch's last op: F >= 0
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        return shortcut + out                       # bare add: f is exactly identity
```
