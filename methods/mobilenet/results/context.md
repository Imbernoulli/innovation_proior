# Context: efficient convolutional networks for on-device vision

## Research question

Image-recognition systems are moving from server-side benchmarks into phones, embedded cameras, robots, and augmented-reality devices. The usual route to higher accuracy has been to make convolutional networks deeper, wider, and more intricately connected. That gives strong ImageNet numbers, but it also creates models with billions of multiply-adds per image and tens or hundreds of millions of parameters. On-device vision has a different constraint: the model must run quickly under tight memory, power, and latency budgets, often without a network round trip.

The design problem is to build a convolutional classifier that keeps most of the accuracy of large models while making inference cheap enough for limited hardware. It also needs a simple way to choose smaller variants for smaller budgets. A useful answer cannot optimize parameter count alone, because a compact model can still be slow. It must account for multiply-adds and for whether the remaining operations map to efficient dense kernels.

## Cost model

A standard convolution maps an input feature map `F` of shape `D_F x D_F x M` to an output `G` of shape `D_F x D_F x N`, using a square kernel `K` of shape `D_K x D_K x M x N`. With stride 1 and padding that preserves spatial size,

```text
G[k,l,n] = sum_{i,j,m} K[i,j,m,n] F[k+i-1,l+j-1,m].
```

The multiply-add cost is

```text
D_K * D_K * M * N * D_F * D_F.
```

The important feature of this expression is multiplicative coupling. Spatial support, input channels, output channels, and feature-map size all multiply one another. A standard convolution also performs two jobs at once: it filters spatial neighborhoods and mixes channels into new features. Any efficient replacement has to preserve enough of both jobs while breaking some of this coupling.

## Prior tools

Several nearby ideas already exist. Inception-style networks use `1 x 1` convolutions for channel projection and factorize some spatial convolutions, such as replacing an `n x n` convolution by `n x 1` followed by `1 x n`. Depthwise separable convolution appears in Sifre's rigid-motion scattering line and is later discussed as an extreme point of the Inception factorization spectrum: per-channel spatial filtering followed by channel mixing. Flattened convolutional networks go further by replacing a 3-D filter with a sequence of one-dimensional filters, showing that aggressive factorization can speed feedforward inference but also imposes a strong rank-style structure.

Other work attacks the deployment problem after training. Pruning, quantization, hashing, product quantization, low-rank tensor expansions, and distillation can compress or accelerate a large trained model. Those approaches are useful, but they do not by themselves give a clean family of architectures that can be trained from scratch at different latency and size budgets.

## Baselines and evaluation

The relevant baselines span both high-accuracy and compact-model regimes. VGG-style networks use deep stacks of standard `3 x 3` convolutions and reach strong accuracy at very high compute and parameter cost. GoogLeNet/Inception models are more efficient and make heavy use of `1 x 1` projections, but their module topology is hand-designed and branchy. SqueezeNet is extremely small in parameters through squeeze/expand modules, yet parameter count is not the same as latency or multiply-add count.

The main evaluation setting is ImageNet classification at input resolutions such as `224`, `192`, `160`, and `128`, reporting top-1 accuracy, multiply-adds, and parameter count. A useful architecture should also act as a transferable backbone for fine-grained classification, geolocation, face attributes or embeddings, and object detection. The training stack available before the architecture choice is standard supervised image classification with convolution, batch normalization, nonlinear activation, pooling, and an optimizer such as RMSProp.

## Code scaffold

The following scaffold exposes the open slot without giving away the block design. It has a standard convolutional stem, a stack of repeated blocks still to be chosen, and a classifier head.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(in_ch, out_ch, kernel, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class CandidateNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = conv_bn_act(3, 32, kernel=3, stride=2, padding=1)
        self.blocks = self._make_blocks(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

    def _make_blocks(self, in_ch):
        raise NotImplementedError

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def train_step(model, images, labels, optimizer):
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```
