## Research question

Convolutional networks have, over a few years, become the dominant tool for large-scale image classification, and the recipe driving the recent gains is to make the network bigger: deeper (more levels of feature abstraction) and wider (more units per level), trained on a large labeled dataset.

Bigger networks carry two costs. More units means more parameters, and more parameters on a dataset with fine-grained, hard-to-distinguish categories raises the overfitting pressure that more labeled data would relieve — and that labeled data is costly to produce, because telling apart two near-identical breeds of dog reliably needs expert annotation. Separately, enlarging a network raises its compute: when two convolutional layers are chained, a uniform increase in the number of filters causes a *quadratic* increase in multiply-add operations. Compute budgets are finite, and there is a growing pull toward models that can run on modest or embedded hardware.

So the setting is: how to obtain the accuracy that a much larger, deeper, and wider network would give, while keeping the inference-time computational budget roughly fixed (on the order of a couple of billion multiply-adds) and the parameter count modest.

## Background

**The standard CNN template.** Since LeNet-5, convolutional networks have followed one shape: a stack of convolutional layers (each optionally followed by a contrast/response normalization and a max-pool) topped by one or more fully-connected layers and a softmax. Variants of this template hold the best results on MNIST, CIFAR, and ImageNet. Convolutions exploit *spatial* locality — each output depends only on a local patch — but are implemented as dense connections within that patch and, in the channel dimension, as full connections. Early convnets used random/sparse connection tables across feature maps to break symmetry; later convnets use dense full connections in the channel dimension, because dense matrix multiplication is what current hardware and tuned numerical libraries do fastest.

**The scaling trend and its tools.** The recent gains came from increasing the number of layers and the size of layers, while using dropout to control the overfitting that the extra parameters bring. Multi-scale processing has an old pedigree too: a neuroscience-inspired model of the primate visual cortex used a series of *fixed* Gabor filters of several sizes to handle multiple scales — a two-layer, hand-set construction.

**The sparsity theory.** A result on learnability of deep representations states, informally, that if the data's distribution is representable by a large, very sparse deep network, then a good network topology can be constructed layer by layer: analyze the correlation statistics of the activations and cluster together units whose outputs are highly correlated; those clusters become the units of the next layer. This resonates with the Hebbian principle — neurons that fire together wire together.

**Hardware and sparse computation.** Non-uniform sparse computation is a poor fit for current infrastructure: even reducing arithmetic operations 100× does not pay off, because the overhead of irregular memory lookups and cache misses dominates, and dense matrix multiply on highly tuned libraries keeps getting faster. The sparse-numerical-linear-algebra literature reports that clustering a sparse matrix into *relatively dense submatrices* yields strong practical performance for sparse matrix multiplication.

**Structural facts about current convnets.** Fully-connected classifier heads dominate the parameter count in the prevailing large convnets, while Network-in-Network has introduced a parameter-free global-average-pooling head as an alternative. Dropout is the standard guard against feature co-adaptation. Relatively *shallow* networks already perform strongly on this task.

## Baselines

**LeNet-5 (LeCun et al., 1989/1998).** The original convolutional template: alternating convolution and subsampling layers, then fully-connected layers. Established weight sharing and local receptive fields. Used sparse/random connection tables across feature maps to break symmetry.

**AlexNet (Krizhevsky et al., 2012).** The deep CNN that won ImageNet 2012 (top-5 error 16.4%). Eight learned layers, ReLU activations, dropout, local response normalization, and GPU-friendly *dense* convolutions; it uses dense feature-map connections to exploit fast parallel compute. Its fully-connected head holds the large majority of its ~60M parameters.

**Layer-enlargement line (Zeiler & Fergus, 2014; Sermanet et al., 2013).** Work that improved on AlexNet largely by increasing layer size and by processing images at multiple scales (and applying the same convnet to localization and detection).

**Network-in-Network (Lin et al., 2013).** Two contributions. (i) Replace a convolution's linear filter with a tiny per-patch multilayer perceptron ("mlpconv"), which, applied convolutionally, is exactly a stack of **1×1 convolutions** followed by ReLU — a learned, nonlinear recombination of channels at each spatial location, dropping into existing pipelines and adding per-location nonlinearity. (ii) Replace the fully-connected head with **global average pooling**: average each final feature map to a single value, producing a class-confidence vector with *no parameters*, which also acts as a structural regularizer against overfitting.

**Region-based detection (Girshick et al., 2014).** For detection, the leading approach proposes category-agnostic regions from low-level cues and then classifies each region with a CNN — establishing the two-stage proposal-plus-classify pipeline that a strong classification backbone plugs into.

## Evaluation settings

The natural yardstick is the ImageNet Large-Scale Visual Recognition Challenge classification task: 1000 leaf-node categories, ~1.2M training images, 50k validation, 100k test, one ground-truth label per image. Performance is reported as top-1 accuracy and, for ranking, top-5 error (correct if the true label is among the five highest-scoring predictions). Inputs are 224×224 RGB crops with per-channel mean subtraction. Standard practice includes data augmentation (random crops of varying size and aspect ratio, photometric distortions, horizontal mirroring), multi-crop and multi-scale testing, and model ensembling, all measured against a fixed inference-compute budget so that accuracy and cost can be traded off explicitly. Training infrastructure of the day is distributed asynchronous SGD with momentum, a fixed step-decay learning-rate schedule, and parameter averaging for the final model.

## Code framework

The primitives below already exist: convolution, pooling, response normalization, ReLU, dropout, a linear layer, channel-wise concatenation, the cross-entropy loss, and an SGD-with-momentum optimizer. What remains open is the repeated local block, the head, and the training setup.

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
    """The repeated building block of the network — the unit that is stacked
    to form the body. Its internal structure is the open design choice."""
    def __init__(self, in_channels, *widths):
        super().__init__()
        # TODO: the local construction.
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
        # TODO: any training setup the design needs.

        # The head: how to turn the final feature maps into class scores.
        self.final_pool = None  # TODO
        self.classifier = None  # TODO

    def forward(self, x):
        x = self.lrn1(self.pool1(self.conv1(x)))
        x = self.pool2(self.lrn2(self.conv3(self.conv2(x))))
        # TODO: run the body, then the head.
        pass


def compute_loss(outputs, target):
    """Cross-entropy on the main output, plus whatever auxiliary terms the
    design introduces."""
    # TODO: combine the main loss with any auxiliary terms the design adds.
    return F.cross_entropy(outputs, target)
```
