## Research question

CIFAR-style ResNets stack the *same* residual block tens to hundreds of times. The single thing being
designed is that block — the unit `H = ReLU(F(x) + shortcut(x))` that the backbone repeats. The goal is
a block that raises test accuracy *across depths and datasets at once*: a shallow net (ResNet-20 on
CIFAR-10), a deep net (ResNet-56 on CIFAR-100), and a very deep net (ResNet-110 on CIFAR-100). The whole
training recipe around the block — initialization, data pipeline, optimizer, schedule, the global pool and
linear classifier, the outer loop — is frozen. So every method on the ladder is one fill of the block, and
nothing else moves; a design that helps at depth 20 must not break at depth 110, and vice versa.

## Prior art before the first rung (the residual-block lineage)

The block the first rung edits is the plain post-activation residual block, and it is itself the
resolution of a line of work on training depth. These are the ancestors the ladder reacts to; the fixed
substrate below is what they converged to.

- **Plain deep stacks (VGG-style, Simonyan & Zisserman 2014).** Uniform 3×3 conv–BN–ReLU layers stacked
  deep. More depth bought accuracy up to a point, then *degraded* — a deeper plain net reaches *higher
  training error* than a shallower one, not from overfitting (training error itself rises) and not from
  vanishing gradients (BN keeps the signals healthy). Gap: depth past ~20 layers is an optimization
  failure, not a capacity one — the solver cannot find the good solution that provably exists.
- **Residual block (He et al. 2015, arXiv:1512.03385).** Reparameterize: let the two-conv stack learn the
  *residual* `F(x) = H(x) − x` and add a parameter-free identity shortcut, `H = ReLU(F(x) + x)`. Now "do
  nothing useful" means `F → 0`, the easiest thing for weight-decayed SGD to reach, so a near-identity
  layer is cheap to learn and degradation reverses. Gap: the *order* inside the block (Conv-BN-ReLU then a
  ReLU **after** the addition) clamps what the shortcut can carry, and at extreme depth the after-add ReLU
  still sits on the identity path.
- **Highway Networks (Srivastava et al. 2015).** Gated skips, `y = H(x)·T(x) + x·C(x)`, LSTM-style learned
  gates. Gap: a gate that drifts closed *removes* the identity path exactly where a deep net needs it, and
  the gates cost parameters — ResNet's fixed identity skip trained better.
- **Batch Normalization (Ioffe & Szegedy 2015).** Per-mini-batch standardization with a learnable scale
  and shift after each conv; it is *why* the plain stack converges at all and is fixed inside the block
  here (every conv is `bias=False` followed by `BatchNorm2d`).

There is room left in the block: the placement of normalization and activation relative to the
convolutions and the shortcut, the scale on the residual branch, whether the branch is even present every
step, and what recalibration (if any) the branch output gets before the addition. That space is the ladder.

## The fixed substrate

A CIFAR-adapted ResNet is frozen and must not be touched. A 3×3/16 stride-1 stem (no 7×7 conv, no max
pool), then three stages — 16 channels on 32×32, 32 on 16×16, 64 on 8×8 — each a run of `CustomBlock`s,
then global average pool into a single linear classifier. Depth is set by the per-stage block counts:
ResNet-20 is `[3,3,3]`, ResNet-56 is `[9,9,9]`, ResNet-110 is `[18,18,18]`. The first block of stages 2
and 3 strides by 2 and doubles the channels, so the shortcut there must match a change in both spatial
size and channel count; every other block sees `stride=1` and equal in/out channels, where the shortcut is
a bare identity. Initialization is fixed (Kaiming `fan_out` on convs, BN weight 1 / bias 0, Kaiming
`fan_in` on the linear head); the data pipeline is fixed (`RandomCrop(32, pad=4)` + `RandomHorizontalFlip`,
per-dataset normalization); the optimizer is fixed (SGD `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`,
cosine annealing over 200 epochs, batch 128). The metric is the **best test accuracy reached during
training**, so a block that trains *faster* or more *stably* — not only to a higher final point — wins.

## The editable interface

Exactly one region is editable — the `CustomBlock` class in `pytorch-vision/custom_residual.py` (the
block lines 31–61). The backbone relies on a fixed contract that every rung's fill must honor:

- constructor signature `CustomBlock(in_planes, planes, stride)`;
- a class attribute `expansion` (`1` for a basic block, `4` for a bottleneck) — the stem's linear head
  reads `64 * expansion`, so changing it changes the whole backbone's channel arithmetic;
- `forward(x)` returns a tensor with `planes * expansion` channels;
- the shortcut handles the dimension mismatch when `stride != 1` or `in_planes != planes * expansion`.

Inside that contract the block is free: the number, kernel size, and grouping of convolutions; the
placement and type of normalization and activation; the shortcut/skip design; channel or spatial
attention; the `expansion` factor; any extra modules. The starting point is the scaffold default — the
plain post-activation basic block. Each rung replaces exactly this class and nothing else.

```python
# EDITABLE region of custom_residual.py (lines 31-61) -- default fill: plain post-activation block
class CustomBlock(nn.Module):
    """Custom residual block for CIFAR ResNets.

    Args:
        in_planes: input channels
        planes: output channels
        stride: spatial stride (1 or 2)

    Must set class attribute `expansion = 1` (or 4 for bottleneck).
    The shortcut dimension must match planes * expansion.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)
```

## Evaluation settings

Three architecture/dataset pairs spanning the depth range, each at seed 42: **ResNet-20 on CIFAR-10**
(shallow, easy dataset), **ResNet-56 on CIFAR-100** (deep, 100-way), and **ResNet-110 on CIFAR-100** (very
deep, 100-way). One metric, higher is better: **best test accuracy (%)** reached at any epoch during the
fixed 200-epoch cosine schedule. A block must clear the interface above and may not change dataset
construction, the optimizer, the global pool, the classifier head, or the outer training loop — only the
block itself.
