**Problem.** The shortcut sweep confirmed: any explicit non-identity shortcut function `h`
hurts, tracking the lambda-product compounding argument rather than representational capacity.
But `h` is not the only operation near the shortcut path. The after-addition operation `f`
(currently ReLU) determines what the *next* unit's identity shortcut actually carries — since
that shortcut is `x_{l+1}` unchanged, and `x_{l+1} = f(y_l)`. If `f` is non-identity, it reaches
the shortcut-connected path one step removed, through every subsequent unit's shortcut input.

**Key idea.** Before attempting to fix `f`, calibrate that this analogy is real: deliberately
make `f` carry the one ingredient every failed shortcut variant shared — a learned per-channel
multiplicative factor — and check whether the same training-error degradation shows up. Batch
normalization is exactly that: `BN(z) = gamma*(z-mean)/sqrt(var+eps) + beta`. Put BN *after* the
addition (before the existing ReLU), so `f` = BN then ReLU instead of plain ReLU.

**Step-1 edit.** Move BN from its current position (after each conv, inside the branch, as
already used) to the merge point: `x_{l+1} = ReLU(BN(x_l + F(x_l)))`. This is deliberately the
wrong direction relative to "make `f` identity" — it makes `f` *further* from identity, on
purpose, as a calibration probe. Test on both ResNet-110 (matching the shortcut sweep) and the
known ResNet-164 bottleneck baseline (5.93%), to check the hazard is structural to the
branch-and-merge topology and not an artifact of one unit shape/depth.

**Prediction.** If the "`f` reaches the shortcut path through the next unit" argument is right,
this should reproduce the shortcut sweep's degradation signature: worse than baseline on both
architectures, elevated training error (not just test error), and specifically slow progress
early in training, since the compounding effect should bite hardest before the network settles
into a regime where the effective per-unit BN factor sits close to 1. Expected severity: in the
range of the sweep's milder failures (high single digits to low teens), not the catastrophic
"fail" cases, since ReLU still follows the BN and nothing here is adversarially initialized the
way an unbiased gate was.

**What this decides.** Confirms or refutes generalizing the shortcut-sweep mechanism from an
explicit `h` to the after-addition `f`, before committing to redesigning `f`.

```python
import torch.nn as nn
import torch.nn.functional as F


class BNAfterAddBlock(nn.Module):
    """Calibration variant: BN moved to the merge point. f = BN -> ReLU
    instead of plain ReLU; everything else (branch, identity/1x1 shortcut)
    unchanged from the original unit."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        # no bn2 here: BN moves to after the addition instead of after conv2
        self.bn_post = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv2(F.relu(self.bn1(self.conv1(x))))
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        y = shortcut + out                       # bare add
        return F.relu(self.bn_post(y))            # f = BN -> ReLU, on the merged signal
```

(The ResNet-164 bottleneck run uses the same substitution — BN moved from after the last 1x1
conv to after the addition — inside the standard 1x1-reduce / 3x3 / 1x1-restore branch.)
