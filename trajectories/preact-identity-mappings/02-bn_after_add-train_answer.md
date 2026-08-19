The shortcut sweep settled one thing cleanly: every explicit non-identity shortcut function `h`
hurt, and the damage tracked how far each variant's effective multiplicative factor sat from
exactly 1 rather than tracking representational capacity — two of the variants could express
identity inside their solution space and still trained to elevated training error, not just
test error. So `h` stays identity. But `h` is not the only operation near the shortcut, and I
want to be careful not to declare the shortcut path "solved" while a second operation reaches
the same path by a route I haven't examined.

The operation I mean is `f`, the ReLU currently applied after the element-wise addition,
`x_{l+1} = f(y_l)` with `y_l = x_l + F(x_l)`. Write two consecutive units back to back: the
second unit's shortcut, now settled as identity, is just `x_{l+1}` passed through unchanged —
which means the second unit's shortcut input *is* whatever `f` produced. If `f` is identity,
the next shortcut carries `y_l` untouched and the clean additive recursion holds exactly. If `f`
is not identity, whatever it does gets baked into the next unit's shortcut before that shortcut
even starts its own journey. So the lambda-product compounding argument that killed every
richer `h` in the sweep is not actually a statement about an explicit shortcut function — it is
a statement about anything sitting on the path connecting a shallow layer to a deep one, and `f`
reaches that path one step removed, through the next unit's identity connection.

I don't want to assume this analogy holds and build directly on it; I want a calibration test
first. The shortcut sweep already tells me what the signature of "an operation with a learned
multiplicative component sitting on this path" looks like: elevated training error, tracking
how far the operation sits from identity. Batch normalization is exactly that kind of operation
— a learned per-channel affine transform, `gamma*(z-mean)/sqrt(var+eps) + beta`, with a
nontrivial multiplicative factor. So the calibration is to deliberately move BN to the merge
point: let `f` be BN followed by the existing ReLU, rather than plain ReLU alone. This is the
wrong direction relative to where I actually want `f` to go — closer to identity, not further —
but that's the point: I want to see the failure mode I'm predicting appear on demand, in a
controlled way, before I trust the argument enough to design a fix around it.

I predict this reproduces the sweep's degradation signature: worse than baseline, elevated
training error rather than only test error, and specifically slow progress at the start of
training, since the compounding effect should bite hardest before the network's weights and
BN's running statistics settle into a regime where each unit's effective factor sits close to 1.
I'd expect the severity to land closer to the sweep's milder failures than its catastrophic
ones, since ReLU still follows the BN here and nothing is adversarially initialized the way an
unbiased gate was. I'm also testing this on the known ResNet-164 bottleneck baseline (5.93%)
alongside ResNet-110, not just the single architecture the sweep used, to check that whatever
happens is a structural property of the branch-and-merge topology rather than an artifact of
one specific unit shape or depth. If the prediction holds on both, that licenses treating "make
`f` an identity" as the real next target. If it doesn't, the "`f` reaches the shortcut path"
argument needs rethinking before I touch `f` any further.

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
