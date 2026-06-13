# Context

The aim is to design an image-classification network that gains accuracy *without* increasing computational complexity or parameter count — building on the repeat-the-same-block simplicity of VGG/ResNet while borrowing the efficiency of Inception's multi-branch processing, using the architectures, grouped-convolution primitives, and training code available in late 2016. The target is ImageNet-1K.

## Research question

Visual recognition has shifted from feature engineering to *network* engineering: features learned by neural nets from large data need little human involvement and transfer across recognition tasks, but the human effort has moved into designing the architectures. And designing architectures gets harder as the number of hyper-parameters grows — width (channels per layer), filter sizes, strides — especially with many layers.

Two design philosophies are on the table, each with a cost. One stacks building blocks of the *same shape* (VGG, inherited by ResNet): this rule sharply reduces the free hyper-parameter choices, exposes depth as an essential dimension, and — being simple — is less likely to over-adapt to a specific dataset, which is why these nets are robust across many vision and even speech/language tasks. The other (Inception) shows that *carefully designed* multi-branch topologies achieve compelling accuracy at low theoretical complexity, via a split-transform-merge pattern — but its modules are hand-customized stage by stage (filter numbers and sizes tailored per transformation), so it is unclear how to adapt it to a new dataset or task without re-doing that design work.

The sharp question: can a network keep the repeat-the-same-block simplicity of VGG/ResNet *and* exploit split-transform-merge in an easy, extensible way — and in doing so, gain accuracy under the *restricted condition of maintaining (or reducing) complexity and model size*? That last clause is what makes the question hard and rare: it is easy to raise accuracy by adding capacity (going deeper or wider), but methods that raise accuracy at fixed FLOPs and fixed parameter count are uncommon. And is there a *dimension* of the design space, besides depth and width, that is worth turning?

## Background

By late 2016 the field has converged on a few load-bearing ideas, and the relevant ones here concern how a building block processes its input.

- **The simple neuron is already a split-transform-merge.** The elementary operation of a fully-connected or convolutional layer is the inner product `Σ_{i=1}^{D} w_i x_i` over a D-channel input `x = [x_1,…,x_D]`. Read structurally, it *splits* `x` into single-channel subspaces `x_i`, *transforms* each by a scaling `w_i x_i`, and *aggregates* by summation.
- **Residual learning.** A residual block computes `y = x + F(x)`, learning a residual `F` on top of a parameter-free identity shortcut; this conditions the optimization so very deep nets train. The efficient instantiation is the **bottleneck**: `1×1 reduce → 3×3 → 1×1 restore`, running the expensive spatial convolution at a reduced channel width. A ResNet can be read as a two-branch network in which one branch is the identity.
- **Split-transform-merge at low complexity (Inception).** An Inception module splits the input into several lower-dimensional embeddings (by 1×1 convolutions), transforms them with specialized filters (3×3, 5×5, …), and merges by concatenation. Its solution space is a strict subspace of that of a single large dense layer on a high-dimensional embedding, so it approaches the representational power of large dense layers at far lower compute.
- **Grouped convolutions.** A convolution can be split into groups: the input and output channels are partitioned into `C` groups and convolutions are performed separately within each group. This dates to the 2012 two-GPU net, where it was introduced to *distribute the model over two GPUs* — an engineering compromise — and is supported by the standard libraries mainly for compatibility. There has been little evidence of using grouped convolutions to *improve* accuracy. The extreme case (number of groups = number of channels) is channel-wise convolution, a component of separable convolutions.
- **Batch Normalization** after convolutions, and He initialization for ReLU nets — the standard training-stabilization tools.
- **An interpretive observation about ResNets** (Veit et al. 2016): because ResNet's behavior is additive (`y = x + F`), a single ResNet can be interpreted as an *ensemble* of many shallower networks — paths of different lengths through the skip connections. This is a diagnostic reading of existing residual nets, not yet a design principle.

A recurring empirical pressure motivates a new design axis: for existing models, *depth and width start to give diminishing returns* — pushing either further yields ever-smaller accuracy gains for ever-larger compute. So a different axis of the design space, if one exists, could be more compute-efficient.

## Baselines

**VGG (Simonyan & Zisserman 2015).** Build very deep nets by stacking blocks of the *same shape* (3×3 conv stacks). *Core idea:* one simple rule (repeat the block; double width when halving the map) collapses the hyper-parameter space and makes depth the dimension to turn. *Gap:* heavy, and offers only depth/width as the knobs to scale.

**ResNet (He et al. 2016).** Stack modules of the same topology, each a residual block `y = x + F(x)` with a bottleneck `1×1 → 3×3 → 1×1`; identity shortcuts (projections only when dimensions change). *Core math:* residual reformulation conditions optimization for great depth; the bottleneck makes deep nets affordable. *Gap (the one this attacks):* it inherits VGG's two scaling knobs — depth and width — and these are exactly where returns are diminishing. A 256→64→64(3×3)→256 bottleneck block has ≈ `256·64 + 3·3·64·64 + 64·256 ≈ 70k` parameters; scaling it means making it deeper or wider.

**Inception / Inception-ResNet (Szegedy et al. 2015/2016).** Multi-branch split-transform-merge modules; Inception-ResNet adds residual connections with branching-and-concatenating inside the residual function. *Core idea:* carefully customized branches give high accuracy at low theoretical complexity. *Gap:* every branch's filter count and size is hand-tailored and the modules differ stage-by-stage, so the number of branches cannot be cleanly isolated as a single factor, and the design does not transfer to new datasets/tasks without redesign.

**Network compression by decomposition (Denton et al. 2014; Jaderberg et al. 2014; Kim et al. 2016; Ioannou et al. 2016).** Reduce redundancy by decomposing convolutions at the spatial and/or channel level; Ioannou et al. use a "root"-patterned network whose branches are realized by grouped convolutions. *Gap:* these trade accuracy *down* for lower complexity (compression), rather than offering an architecture with stronger representational power at the same complexity.

**Ensembling (averaging independently trained nets).** A reliable accuracy boost, widely used in competitions. *Gap / contrast:* an ensemble's members are trained *independently*; a building block that aggregates several jointly-trained transformations is not the same thing, and should not be conflated with ensembling.

## Evaluation settings

- **ImageNet-1K classification, 1000 classes.** ~1.28M training images. Metric: top-1 (and top-5) error on the validation set. Ablations use the error on a single 224×224 center crop from an image whose shorter side is 256.
- **Cost metrics, held fixed.** Computational complexity in FLOPs (multiply-adds) and parameter count — these are treated as the *inherent capacity* of a model and are held (approximately) constant when comparing design axes, so that an accuracy change is attributable to the architecture rather than to added capacity. A controlled comparison pits a baseline block against a modified block at matched FLOPs and matched parameters.
- **The decisive diagnostic.** Training-error curves alongside validation error: if a modified block lowers *training* error, the gain is from stronger representation, not from regularization.
- **Training protocol (the mature recipe to hold fixed).** 224×224 random crops with scale and aspect-ratio augmentation; SGD, mini-batch 256 across 8 GPUs, momentum 0.9, weight decay 1e-4; learning rate starting at 0.1, divided by 10 three times; BN after each convolution; He initialization. Downsampling at stage transitions by stride-2 in the 3×3 layer of the first block of the stage; identity shortcuts except dimension-increasing projections.
- **Larger yardsticks (already established).** A larger ImageNet-5K set and the COCO detection benchmark, as transfer tests of better classification features.

## Code framework

The available code is a residual-network image-classification harness (following the standard ResNet training code), with grouped convolution available as a primitive in the conv library. It supplies convolution (with a `groups` argument), batch normalization, ReLU, an SGD-with-momentum optimizer with step decay, a cross-entropy loss, and the scale/aspect-ratio augmentation pipeline. The scaffold is the ResNet harness with one empty slot: the residual block's transformation, and how the per-stage template is parameterized.

```python
import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    # the conv library exposes a `groups` argument: input and output channels
    # are partitioned into `groups` groups, convolved separately within each.
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


class Block(nn.Module):
    # TODO: the residual block we'll design. Known going in (inherited from the
    # residual/bottleneck baseline): it is a residual function added onto an
    # identity (or, at stage transitions, a projection) shortcut, of bottleneck
    # shape (1x1 reduce -> 3x3 -> 1x1 restore). What the transformation inside
    # the block is, and how the per-stage template is parameterized, is the open slot.
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class Net(nn.Module):
    # Fixed by the VGG/ResNet template rules: a 7x7/64 stride-2 stem + maxpool,
    # then four stages that repeat `Block` (same map size -> same width; halve
    # the map -> double the width), global average pool, single FC. The block
    # and its parameterization are the open slot.
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


# --- training harness ---
model = Net(Block, [3, 4, 6, 3])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[...], gamma=0.1)
# scale + aspect-ratio augmentation; 224x224 random crop; SGD batch 256 on 8 GPUs.
```
