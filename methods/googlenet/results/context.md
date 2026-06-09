## Research question

Convolutional networks have, over a few years, become the dominant tool for large-scale image classification, and the recipe driving the gains is blunt: make the network bigger. Bigger means deeper (more levels of feature abstraction) and wider (more units per level), and with a large labeled dataset this is the easy, safe way to a better model.

The problem is that "bigger" is expensive in two ways that both scale badly. First, more units means more parameters, and more parameters on a dataset with fine-grained, hard-to-distinguish categories invites overfitting — and the labeled data that would cure overfitting is exactly what is costly to produce, because telling apart two near-identical breeds of dog reliably needs expert annotation. Second, naively enlarging a network wastes computation: when two convolutional layers are chained, a uniform increase in the number of filters causes a *quadratic* increase in multiply-add operations, and if much of that added capacity ends up with weights near zero, the compute is simply thrown away. Compute budgets are finite, and there is a growing pull toward models that can actually run on modest or embedded hardware.

So the precise goal: obtain the accuracy that a much larger, deeper, and wider network would give, while keeping the inference-time computational budget roughly fixed (on the order of a couple of billion multiply-adds) and the parameter count modest. A solution would have to *distribute computation efficiently* — spend capacity where it buys accuracy and nowhere else — rather than scaling every layer uniformly.

## Background

**The standard CNN template.** Since LeNet-5, convolutional networks have followed one shape: a stack of convolutional layers (each optionally followed by a contrast/response normalization and a max-pool) topped by one or more fully-connected layers and a softmax. Variants of this template hold the best results on MNIST, CIFAR, and ImageNet. Convolutions already exploit *spatial* sparsity — each output depends only on a local patch — but they are implemented as dense connections within that patch and, in the channel dimension, as full connections. Early convnets used random/sparse connection tables across feature maps to break symmetry; the field then reverted to dense full connections in the channel dimension precisely because dense matrix multiplication is what current hardware and tuned numerical libraries do fastest.

**The scaling trend and its tools.** The recent gains came from increasing the number of layers and the size of layers, while leaning on dropout to control the overfitting that the extra parameters bring. Multi-scale processing has an old pedigree too: a neuroscience-inspired model of the primate visual cortex used a series of *fixed* Gabor filters of several sizes to handle multiple scales — a two-layer, hand-set construction, but the multi-scale intuition is the same one a learned architecture might want to capture.

**The sparsity argument.** There is a theoretical reason to believe the fundamental fix is *sparse* connectivity, not bigger dense layers. A result on learnability of deep representations states, informally, that if the data's distribution is representable by a large, very sparse deep network, then a good network topology can be constructed layer by layer: analyze the correlation statistics of the activations and cluster together units whose outputs are highly correlated; those clusters become the units of the next layer. This resonates with the Hebbian principle — neurons that fire together wire together — which suggests the prescription is useful in practice even when the strict mathematical preconditions don't hold exactly.

**The hardware wall that blocks naive sparsity.** Despite the theory, non-uniform sparse computation is a poor fit for current infrastructure: even reducing arithmetic operations 100× does not pay off, because the overhead of irregular memory lookups and cache misses dominates, and dense matrix multiply on highly tuned libraries keeps getting faster. So a literal sparse network is impractical. The sparse-numerical-linear-algebra literature is the nearest adjacent fact: clustering a sparse matrix into *relatively dense submatrices* yields strong practical performance for sparse matrix multiplication. The standing tension is that the theory argues for sparsity while the hardware rewards only dense, regular computation, and prior architectures sit on one horn or the other.

**Diagnostic facts that frame the design.** The useful facts before designing a new block are structural rather than benchmark numbers. Fully-connected classifier heads dominate the parameter count in the prevailing large convnets, while Network-in-Network has already made a parameter-free global-average-pooling head a plausible alternative. Dropout remains the standard guard against feature co-adaptation. And the strong performance of relatively *shallow* networks on this task is itself a clue: the features in the middle of a deep network ought already to be quite discriminative.

## Baselines

**LeNet-5 (LeCun et al., 1989/1998).** The original convolutional template: alternating convolution and subsampling layers, then fully-connected layers. Established weight sharing and local receptive fields. Used sparse/random connection tables across feature maps to break symmetry. Limitation as a yardstick: tiny by modern standards and single-scale; not built for the parameter/compute pressures of ImageNet-scale problems.

**AlexNet (Krizhevsky et al., 2012).** The deep CNN that won ImageNet 2012 (top-5 error 16.4%). Eight learned layers, ReLU activations, dropout, local response normalization, and GPU-friendly *dense* convolutions; it deliberately abandoned sparse feature-map connection tables to exploit fast parallel dense compute. Its fully-connected head holds the large majority of its ~60M parameters. Gap it leaves: the sheer parameter count (overfitting pressure, memory) and the fact that scaling such a design uniformly drives compute up quadratically — most of the cost living in layers that may not need it.

**Layer-enlargement line (Zeiler & Fergus, 2014; Sermanet et al., 2013).** Work that improved on AlexNet largely by increasing layer size and by processing images at multiple scales (and applying the same convnet to localization and detection). Confirms that width and multi-scale help, but continues to scale dense layers, so the compute/parameter problem only grows.

**Network-in-Network (Lin et al., 2013).** The most direct ancestor. Two contributions. (i) Replace a convolution's linear filter with a tiny per-patch multilayer perceptron ("mlpconv"), which, applied convolutionally, is exactly a stack of **1×1 convolutions** followed by ReLU — a learned, nonlinear recombination of channels at each spatial location, dropping cleanly into existing pipelines and raising representational power. (ii) Replace the fully-connected head with **global average pooling**: average each final feature map to a single value, producing a class-confidence vector with *no parameters*, which also acts as a structural regularizer against overfitting. Gap it leaves: it is still a single-scale stack; its 1×1 convolutions are introduced only to add per-location nonlinearity, and there is no explicit story for keeping a large, multi-scale network within a fixed budget.

**Region-based detection (Girshick et al., 2014).** For detection, the leading approach proposes category-agnostic regions from low-level cues and then classifies each region with a CNN — establishing the two-stage proposal-plus-classify pipeline that a strong classification backbone would plug into. (Relevant as the downstream consumer of a better backbone, not as a classification baseline.)

## Evaluation settings

The natural yardstick is the ImageNet Large-Scale Visual Recognition Challenge classification task: 1000 leaf-node categories, ~1.2M training images, 50k validation, 100k test, one ground-truth label per image. Performance is reported as top-1 accuracy and, for ranking, top-5 error (correct if the true label is among the five highest-scoring predictions). Inputs are 224×224 RGB crops with per-channel mean subtraction. Standard practice includes data augmentation (random crops of varying size and aspect ratio, photometric distortions, horizontal mirroring), multi-crop and multi-scale testing, and model ensembling, all measured against a fixed inference-compute budget so that accuracy and cost can be traded off explicitly. Training infrastructure of the day is distributed asynchronous SGD with momentum, a fixed step-decay learning-rate schedule, and parameter averaging for the final model.

## Code framework

The primitives below already exist: convolution, pooling, response normalization, ReLU, dropout, a linear layer, channel-wise concatenation, the cross-entropy loss, and an SGD-with-momentum optimizer. What remains open is the repeated local block that will spend computation efficiently, the head, and whether any training-time auxiliary signal is needed.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvUnit(nn.Module):
    """Convolution followed by a rectified linear activation — the basic
    primitive every layer is built from."""
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)


class FeatureBlock(nn.Module):
    """The repeated building block of the network — the unit that will be
    stacked spatially to form the body. Its internal structure (how it
    spends its compute, what it connects to what) is the open design
    problem."""
    def __init__(self, in_channels, *widths):
        super().__init__()
        # TODO: the local construction that this whole effort is about.
        pass

    def forward(self, x):
        # TODO: produce this block's output feature map from x.
        pass


class Net(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # A conventional stem: a few plain convs, pooling, response norm.
        self.conv1 = ConvUnit(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.lrn1 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=1.0)
        self.conv2 = ConvUnit(64, 64, kernel_size=1)
        self.conv3 = ConvUnit(64, 192, kernel_size=3, padding=1)
        self.lrn2 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=1.0)
        self.pool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # The body: a stack of the building block, with pooling to reduce
        # resolution between stages. Widths/arrangement TODO once the block
        # is designed.
        self.body = nn.ModuleList()  # TODO: fill with FeatureBlock(...) stages
        # TODO: whatever training-time machinery (if any) the design turns
        # out to need to make a deep stack trainable.

        # The head: how to turn the final feature maps into class scores
        # without an overfitting-prone fully-connected stack is itself a
        # choice to make.
        self.final_pool = None  # TODO
        self.classifier = None  # TODO

    def forward(self, x):
        x = self.lrn1(self.pool1(self.conv1(x)))
        x = self.pool2(self.lrn2(self.conv3(self.conv2(x))))
        # TODO: run the body (plus any training-time machinery), then the head.
        pass


def compute_loss(outputs, target):
    """Cross-entropy on the main output, plus whatever auxiliary terms the
    design introduces."""
    # TODO: combine the main loss with any auxiliary terms the design adds.
    return F.cross_entropy(outputs, target)
```
