## Research question

CIFAR-style ResNets repeat the same residual block tens to hundreds of times. The design task is that single block — the unit `H = ReLU(F(x) + shortcut(x))` that the backbone repeats. The goal is a block that raises test accuracy across depths and datasets at once: ResNet-20 on CIFAR-10, ResNet-56 on CIFAR-100, and ResNet-110 on CIFAR-100. The surrounding training recipe — initialization, data augmentation, optimizer, schedule, global pooling, linear classifier, outer loop — is frozen. Each candidate changes only the block; a design that helps at depth 20 must not break at depth 110, and vice versa.

## Prior art / Background / Baselines

The starting block is the plain post-activation residual block. The relevant prior work is:

- **Plain deep stacks (VGG-style, Simonyan & Zisserman 2014).** Core idea: a uniform stack of 3×3 conv–BN–ReLU layers. Gap: beyond ~20 layers, training error rises with depth despite BN, so extra depth is an optimization failure, not a capacity shortage.
- **Residual block (He et al. 2015).** Core idea: let a two-conv stack learn the residual `F(x) = H(x) − x` and add a parameter-free identity shortcut, `H = ReLU(F(x) + x)`, so "do nothing" means `F → 0`. Gap: at 110-layer scale, the original block still under-optimizes relative to shallower networks; depth scaling is not fully closed.
- **Highway Networks (Srivastava et al. 2015).** Core idea: gated skip paths, `y = H(x)·T(x) + x·C(x)`, with learned LSTM-style gates. Gap: gates can suppress the skip path where depth needs it most, and the gate parameters make training less reliable than a fixed identity skip.
- **Batch Normalization (Ioffe & Szegedy 2015).** Normalizes activations per mini-batch with learnable scale and shift; it is fixed inside every block here as `bias=False` convolutions followed by `BatchNorm2d`.

## Fixed substrate / Code framework

The CIFAR-adapted ResNet backbone is frozen. A 3×3/16 stride-1 stem feeds three stages — 16 channels on 32×32, 32 on 16×16, 64 on 8×8 — each a run of `CustomBlock`s, then global average pooling and a single linear classifier. Depths are set by per-stage block counts: `[3,3,3]` for ResNet-20, `[9,9,9]` for ResNet-56, `[18,18,18]` for ResNet-110. The first block of stages 2 and 3 strides by 2 and doubles channels, so its shortcut must match spatial and channel changes; all other blocks use `stride=1` and identity shortcuts.

Initialization is fixed: Kaiming `fan_out` for convs, BN weight 1 / bias 0, Kaiming `fan_in` for the linear head. The data pipeline is fixed: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip` and per-dataset normalization. The optimizer is fixed: SGD `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`, cosine annealing over 200 epochs, batch size 128. The metric is the **best test accuracy reached during training**, so a block that trains faster or more stably wins.

## Editable interface

Only the `CustomBlock` class in `pytorch-vision/custom_residual.py` is editable. The backbone requires:

- constructor signature `CustomBlock(in_planes, planes, stride)`;
- class attribute `expansion` (`1` for basic block, `4` for bottleneck), which sets the linear head input dimension to `64 * expansion`;
- `forward(x)` returns a tensor with `planes * expansion` channels;
- the shortcut handles dimension mismatch when `stride != 1` or `in_planes != planes * expansion`.

Inside that contract the block is free to use any submodules and topology; the default fill is the plain post-activation basic block below.

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

Three architecture/dataset pairs at seed 42: **ResNet-20 on CIFAR-10** (shallow, 10-way), **ResNet-56 on CIFAR-100** (deep, 100-way), and **ResNet-110 on CIFAR-100** (very deep, 100-way). One metric, higher is better: **best test accuracy (%)** reached at any epoch during the fixed 200-epoch cosine schedule. Candidates must respect the interface above and may not change dataset construction, optimizer, global pooling, classifier head, or the outer training loop — only the block.
