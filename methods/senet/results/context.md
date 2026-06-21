# Context

## Research question

The central computational unit of a convolutional network is the convolution. At each layer a bank of filters slides over the input feature maps and, at every spatial location, fuses information across both space and channels within a *local* receptive field. Concretely, the c-th output map is produced by summing convolutions of every input channel: each filter mixes the channels together with a fixed set of learned weights, and that mixing is the same regardless of which image is being processed and regardless of what the rest of the image looks like outside the filter's small spatial window.

The question is whether the representational power of a convolutional network can be improved by giving it an explicit, cheap, input-dependent mechanism to model the interdependencies between channels, using global information about an image to selectively emphasise or suppress individual channels.

## Background

The dominant route to stronger image representations has been to go deeper and to enrich the spatial side of the convolution. Increasing depth, as in VGGNets (Simonyan & Zisserman 2015) and the Inception family (Szegedy et al. 2015), markedly improves the quality of learned features. Batch Normalization (Ioffe & Szegedy 2015) re-normalises layer inputs and stabilises the optimisation of deep networks, producing smoother loss surfaces (Santurkar et al. 2018) and making very deep training practical. Residual networks (He et al. 2016) add identity skip connections so that a block need only learn a residual on top of its input, allowing networks of unprecedented depth to train; a pre-activation variant (He et al. 2016b) further eases optimisation.

A second, related line reshapes the *functional form* of the unit inside a block. Grouped convolutions partition channels into groups to increase the cardinality of the learned transform (Xie et al. 2017); multi-branch modules (Inception) combine operators of several receptive-field sizes; 1x1 convolutions (Lin et al. 2014) and depthwise-separable factorisations (Chollet 2017) remap channels as new linear combinations. A common thread is that cross-channel structure is modelled as a *composition of instance-agnostic functions over local receptive fields*, frequently with the explicit aim of reducing computation.

A third relevant idea is *gating*. Highway networks (Srivastava et al. 2015) place learned, data-dependent gates on shortcut connections to regulate how information flows through a deep network — a precedent for letting the network multiplicatively modulate its own signal with values it computes on the fly.

A fourth is *attention*, understood as biasing the allocation of computation toward the most informative parts of a signal (Itti & Koch 1998; Mnih et al. 2014; Vaswani et al. 2017). Attention has been used for sequence learning, localisation, and captioning, typically inserted after layers representing higher-level abstractions. Within vision, spatial attention has been folded into the architecture (e.g. spatial transformer modules, Jaderberg et al. 2015), and trunk-and-mask designs (Wang et al. 2017) compute soft attention masks with auxiliary hourglass branches inserted between residual stages.

Finally, classical feature-engineering work — spatial pyramid pooling, Fisher vectors — establishes that *globally aggregated statistics* of local descriptors are highly expressive for whole-image recognition.

## Baselines

- **Plain deep CNN (VGG / Inception).** Stacks of conv + nonlinearity (with batch norm), optionally multi-branch. Channel relationships are whatever the spatial filters implicitly learn; the transform is fixed at test time and uses no explicit global channel statistics.

- **Residual network.** A block computes y = x + F(x), where F is a small stack of convolutions (for deeper nets a 1x1-reduce → 3x3 → 1x1-expand bottleneck), each followed by batch norm and ReLU, and the result is added to an identity (or projection) shortcut, then a final ReLU. Skip connections make depth trainable and give the strongest backbones of the time.

- **Highway network.** Replaces the plain identity carry with learned gates T(x), H(x): y = H(x)·T(x) + x·(1−T(x)). Demonstrates useful data-dependent multiplicative gating that regulates flow along the depth dimension.

- **Trunk-and-mask spatial attention.** Auxiliary hourglass branches produce soft attention masks that modulate the trunk features between stages. Captures input-dependent attention, oriented toward spatial (and combined spatial/channel) masking.

## Evaluation settings

The natural yardstick is large-scale image classification on ImageNet (ILSVRC 2012): ~1.28M training images over 1000 classes, ~50K validation images, reporting top-1 and top-5 error on single-centre/standard crops at 224x224. Standard training is SGD with momentum, stepwise (or plateau-triggered) learning-rate decay, scale/aspect-ratio and horizontal-flip augmentation, and optionally label smoothing. Model cost is reported as parameter count and GFLOPs for a single 224x224 forward pass, plus wall-clock train/inference timings on GPU and CPU. Backbones such as ResNet-50 / ResNet-101 / ResNeXt and Inception serve as the reference architectures into which any new unit would be inserted, so that improvement is measured at matched or near-matched depth and cost. Transfer to other datasets and tasks is used to check that gains are not dataset-specific.

## Code framework

The primitives below already exist: convolution, batch norm, ReLU, global pooling, fully-connected layers, the residual block, and a standard ImageNet training loop. The contribution will be a single new module dropped into the residual branch, plus the small change to the block that calls it. Everything method-specific is left as a stub.

```python
import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ChannelRecalibration(nn.Module):
    """A lightweight, input-dependent reweighting of feature-map channels
    using global context. To be designed."""
    def __init__(self, channels):
        super().__init__()
        # TODO: build the module.
        pass

    def forward(self, u):
        # TODO: return the recalibrated feature maps (same shape as u).
        raise NotImplementedError


class Bottleneck(nn.Module):
    """Standard residual bottleneck: 1x1 reduce -> 3x3 -> 1x1 expand,
    BN + ReLU between, added to the shortcut."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # TODO: a module that recalibrates the residual-branch channels
        #       before the addition.
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # TODO: apply channel recalibration to `out` here, before the add.
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


def train_imagenet(model, loader, epochs):
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for images, targets in loader:
            opt.zero_grad()
            crit(model(images), targets).backward()
            opt.step()
```
