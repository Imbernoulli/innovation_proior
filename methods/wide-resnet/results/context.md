# Context

The aim is to make residual networks both more accurate and far faster to train than the prevailing strategy of pushing depth ever higher — by re-examining the residual block's structure (its convolution layout, number of layers, and width) and its regularization, on CIFAR/SVHN/ImageNet, using the residual-net architectures, normalization, dropout, and training code available in mid-2016.

## Research question

Deep residual networks scale to thousands of layers and keep improving — but at a brutal exchange rate: each additional fraction of a percent of accuracy costs roughly *doubling* the number of layers. So training the very deep residual networks that hold the state of the art is extremely slow. And there is a structural reason to suspect that most of that depth is wasted: the identity-mapping shortcut that makes great depth trainable is, at the same time, a weakness — as gradient flows backward, nothing *forces* it to pass through a block's residual weights, so a block can simply pass its input along and learn almost nothing. The likely consequence is that only a handful of blocks learn useful representations while the rest contribute little — a phenomenon named *diminishing feature reuse*.

So far the study of residual networks has concentrated on just two axes: the order of activations inside a block, and depth. The precise question here goes beyond those: across the much richer space of residual-block designs — how many convolution layers per block, what kernels, and how wide each block is — what structure gives the best accuracy per unit of training time? And once a block design is fixed, how should the resulting network be regularized?

## Background

By mid-2016 the residual network is the dominant image-recognition backbone, and several facts and tools frame any redesign of it.

- **The residual block.** A residual unit computes `x_{l+1} = x_l + F(x_l, W_l)`, learning a residual `F` on a parameter-free identity shortcut; stacking these lets very deep nets train. The two block shapes in use are the **basic** block (two stacked 3×3 convolutions) and the **bottleneck** block (`1×1 reduce → 3×3 → 1×1 expand`), the latter introduced specifically to make blocks *thinner* so that more layers fit in a fixed budget.
- **Pre-activation ordering.** Reordering a block from `conv → BN → ReLU` to `BN → ReLU → conv` (pre-activation, He et al. 2016) trains faster and reaches better accuracy, and is the order used for very deep nets.
- **Diminishing feature reuse, made concrete.** Highway networks (Srivastava et al.) — gated residual links whose gates are learned — first formulated the worry that deep nets under-use their blocks. Stochastic depth (Huang et al.) addresses it by *randomly dropping whole residual blocks during training*; this is a special case of dropout in which each block has an identity scalar weight subjected to dropout. That this *works* — and even improves very deep nets — is direct evidence that many blocks in a deep residual net are barely contributing.
- **Regularization tools and their trade-offs.** Dropout (Srivastava et al. 2014) randomly zeroes unit outputs to prevent co-adaptation; it was historically applied to the parameter-heavy top layers. Batch Normalization (Ioffe & Szegedy 2015) normalizes activations to reduce internal covariate shift, also acts as a regularizer, and was shown to beat dropout — so dropout largely fell out of use in these nets. But BN's regularization leans on heavy data augmentation, which is not always available (e.g. SVHN, where the diagnostic is that without augmentation the training loss drops to very low values — BN overfits).
- **The pro-depth prior, and the counter-evidence.** Circuit-complexity arguments say shallow circuits can need exponentially more components than deep ones, which is part of why the residual-net designers made their nets as *thin* as possible to spend the budget on depth. But almost every successful pre-residual architecture — VGG, Inception — was much *wider* than thin residual nets, which is at least suggestive that width was prematurely abandoned.
- **A hardware fact.** Thin, deep nets built from many small kernels run against the grain of GPUs, whose efficiency comes from parallel computation on large tensors; a sequential stack of small convolutions under-uses the hardware. Parameters and compute are *quadratic* in a width multiplier but only *linear* in the number of blocks.

## Baselines

**Thin deep ResNet / pre-activation ResNet (He et al. 2015, 2016).** Residual blocks `x + F(x)` stacked very deep and made deliberately thin (basic or bottleneck), with pre-activation `BN → ReLU → conv` ordering. *Core idea:* depth is the dimension to scale; thinness keeps parameters down. *Gap (what this attacks):* the accuracy-per-layer exchange rate is terrible (≈doubling layers per fractional gain), training is slow, and diminishing feature reuse means much of the depth is idle. Thousand-layer nets (ResNet-1001, ResNet-1202) are the extreme case — heavy and slow, and not clearly better than far shallower nets of comparable parameter count.

**Highway networks (Srivastava et al.).** Residual-like links that are *gated*, with learned gate weights deciding how much signal carries through. *Core idea:* learnable skip gating to train very deep nets. *Gap:* the gates add parameters and can throttle the skip; and Highway named diminishing feature reuse without resolving it.

**Stochastic depth (Huang et al.).** Train a very deep net while randomly dropping entire residual blocks (each block scaled by a Bernoulli identity weight). *Core idea:* a per-block dropout that shortens the effective network during training. *Role:* less a competitor than a *probe* — its success confirms that many blocks are nearly redundant, which is exactly the motivation for not building so many of them.

**Dropout on top layers (the classic usage).** Zero unit outputs with some probability in the large fully-connected layers. *Gap:* designed for the parameter-heavy FC head of older nets; in residual nets it was displaced by BN, and an attempt to place dropout in the *identity* path of a residual block was reported to *hurt*. Where dropout should go inside a residual block was an open question.

**Wider classical architectures (VGG, Inception).** Much wider per-layer than thin residual nets, at far fewer layers. *Role:* existence proof that width is a viable axis; a width-8–10 residual net of a couple dozen layers is comparable to VGG in width, depth, and parameter count.

## Evaluation settings

- **CIFAR-10 / CIFAR-100.** 32×32 color images, 10 / 100 classes, 50,000 train / 10,000 test. Data augmentation: horizontal flips and random crops from images reflect-padded by 4 pixels per side (no heavy augmentation). Preprocessing: mean/std normalization (to compare directly with prior residual-net work) or ZCA whitening.
- **SVHN.** ~600,000 Street View House Number digit images; a harder real-world problem. No preprocessing beyond dividing by 255 into [0,1]; *no* data augmentation — which is exactly the regime where BN's regularization is weakest.
- **ImageNet.** 1000-class classification; top-1 and top-5 error.
- **Metrics and protocol.** Classification test error (CIFAR/SVHN report median over 5 runs); top-1/top-5 on ImageNet. Wall-clock training time per epoch / per minibatch (cuDNN, single GPU) is a first-class metric, since the whole point is accuracy-per-training-time. Comparisons across block designs are made at *matched parameter count* so an accuracy change reflects the structure, not added capacity. A decisive diagnostic is the pair of training-loss and test-error curves (whether width buys a true representational gain or merely regularizes).
- **A network-family parameterization.** A residual net of three groups, each of `N` blocks, with a deepening factor `l` (convolutions per block) and a widening factor `k` (feature-plane multiplier); the original thin net corresponds to `k = 1`.

## Code framework

The available code is a residual-network image-classification harness with pre-activation blocks. The libraries supply convolution, batch normalization, ReLU, dropout, an SGD-with-Nesterov-momentum optimizer with step decay, and a cross-entropy loss. The scaffold is the harness with the residual block parameterized by free width/depth factors but its internal structure and regularization left as the open slot.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    # TODO: the residual block we'll design. Known going in: pre-activation
    # (BN -> ReLU -> conv) residual function F added onto an identity (or 1x1
    # projection when the channel count changes) shortcut. Open: how many
    # convolution layers it has, their kernel sizes, the width of its feature
    # planes, and where (if anywhere) a regularizer goes inside it.
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class Net(nn.Module):
    # Fixed by the residual template: a single 3x3 stem at width 16, then three
    # groups of `n` blocks at widths (16*k, 32*k, 64*k); the map is halved (and
    # width set) at the start of groups 2 and 3; a final BN-ReLU, global average
    # pool, and a classification FC. The width factor k and depth (n per group)
    # are free knobs; the block is the open slot.
    def __init__(self, depth, k, num_classes=10):
        super().__init__()
        assert (depth - 4) % 6 == 0, 'depth must be 6n+4'
        n = (depth - 4) // 6
        widths = [16 * k, 32 * k, 64 * k]
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.group1 = self._make_group(16, widths[0], n, stride=1)
        self.group2 = self._make_group(widths[0], widths[1], n, stride=2)
        self.group3 = self._make_group(widths[1], widths[2], n, stride=2)
        self.bn = nn.BatchNorm2d(widths[2])
        self.fc = nn.Linear(widths[2], num_classes)

    def _make_group(self, in_planes, out_planes, n, stride):
        # TODO: stack n blocks; first block carries the stride / width change
        raise NotImplementedError

    def forward(self, x):
        x = self.conv1(x)
        x = self.group1(x); x = self.group2(x); x = self.group3(x)
        x = F.relu(self.bn(x))
        x = F.avg_pool2d(x, 8).flatten(1)
        return self.fc(x)


# --- training harness ---
model = Net(depth=28, k=10, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                            weight_decay=5e-4, nesterov=True, dampening=0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
# CIFAR augmentation: random crop from 4-pixel reflect-pad + horizontal flip; mean/std norm.
```
