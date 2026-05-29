# DenseNet: the method distilled

## Problem

Very deep convolutional networks are hard to train: as the input signal flows forward and the
gradient flows backward through many nonlinear layers, both attenuate, and a plain deep stack
shows the degradation problem (higher *training* error than a shallower net — an optimization
failure, not overfitting). Residual networks fix the flow with an additive identity shortcut, but
they merge preserved and newly-computed features by summation and give every layer its own
full-width weights, much of which is redundant. The goal: a connectivity pattern that maximizes
forward/backward flow *and* uses parameters efficiently.

## Key idea

Connect **every layer to every subsequent layer** within a region, and combine by
**concatenation** instead of summation. Layer ℓ receives the concatenation of all preceding
feature-maps,

    x_ℓ = H_ℓ( [x_0, x_1, …, x_{ℓ−1}] ),

so an L-layer block has L(L+1)/2 direct connections instead of L. Because every earlier
feature-map is preserved verbatim and directly readable by every later layer in the same block, no
layer needs to re-produce or re-carry existing features inside that block. Across blocks, transition
layers pass a mixed, spatially downsampled state onward. The concatenated stack is a shared "global
state"; each layer only **adds** a small fixed number of new feature-maps to it.

This is the **Dense Convolutional Network (DenseNet)**. It explicitly separates preserved
information (the untouched concatenated history) from added information (the new maps), which is
exactly what summation conflates.

## The architecture

- **Composite function H_ℓ**: pre-activation BN → ReLU → Conv(3×3), producing **k** feature-maps.
- **Growth rate k**: the number of new maps each layer contributes (e.g. k = 12 on CIFAR, 32 on
  ImageNet). Layer ℓ has k_0 + k(ℓ−1) input channels, where k_0 is the channel count entering the
  block. A 3×3 convolution from c_in to c_out has about 9c_in c_out parameters, so an L-layer block
  without bottlenecks costs

      9k · Σ_{ℓ=1}^{L}(k_0 + k(ℓ−1))
      = 9k · (Lk_0 + kL(L−1)/2)
      = O(k²L²).

  A constant-width W network costs O(W²L). The comparison is k²L² versus W²L, and the small
  coefficient matters: for k = 12 and W = 256, k²/W² = 144/65,536 ≈ 1/455, on the order of 1/400.
- **Dense blocks + transition layers**: concatenation needs equal spatial size, so the network is
  split into dense blocks (full dense connectivity at one resolution) joined by transition layers
  (BN → 1×1 Conv → 2×2 average pool) that downsample.
- **Bottleneck (-B)**: a layer's input grows with depth while its output is fixed at k, so insert a
  1×1 Conv before the 3×3 that squeezes the input to **4k** channels; H becomes
  BN-ReLU-Conv(1×1, →4k) - BN-ReLU-Conv(3×3, →k). The 3×3's cost no longer grows with depth.
- **Compression (-C)**: a transition emits ⌊θ·m⌋ of its m input maps. θ = 1 keeps all channels;
  θ = 0.5 deliberately halves the channel count between blocks after a 1×1 mixing layer selects and
  recombines the block output.
  Bottleneck + compression together = **DenseNet-BC**.
- **Stem and head**: ImageNet uses a 7×7 stride-2 conv + 3×3 stride-2 maxpool stem and 4 blocks;
  CIFAR uses a single 3×3 conv stem and 3 blocks. A final BN, global average pool, and a linear
  classifier sit on top; the single shared loss gives every layer short supervision paths through
  the block-local concatenations and the transition layers.

Why training is easy: inside a block, an early feature-map is literally a channel slice of every
later layer's input, so signal and gradient avoid the full stack of intervening transformations.
Across blocks, only the transition layers separate resolutions.

Standard training: SGD with Nesterov momentum 0.9, weight decay 1e-4, He initialization,
step-decayed learning rate from 0.1; dropout after convolutions on datasets without augmentation.

## Working code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    """H_ell: 1x1 bottleneck (-> 4k) then 3x3 conv (-> k new feature-maps)."""
    def __init__(self, num_input_features, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, prev_features):
        x = torch.cat(prev_features, 1)               # read all preceding maps
        x = self.conv1(self.relu1(self.norm1(x)))     # 1x1 -> 4k
        x = self.conv2(self.relu2(self.norm2(x)))     # 3x3 -> k
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x


class DenseBlock(nn.ModuleDict):
    """Dense connectivity at one resolution; output width = num_input + num_layers*k."""
    def __init__(self, num_layers, num_input_features, growth_rate,
                 bn_size=4, drop_rate=0.0):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate,
                               growth_rate, bn_size, drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for _, layer in self.items():
            features.append(layer(features))
        return torch.cat(features, 1)


class Transition(nn.Sequential):
    """1x1 conv (compress by theta) + 2x2 average pool (downsample)."""
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Module):
    """DenseNet-BC. block_config (6,12,24,16) with growth_rate 32 = DenseNet-121."""
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.0,
                 compression=0.5, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential()
        self.features.add_module("conv0",
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False))
        self.features.add_module("norm0", nn.BatchNorm2d(num_init_features))
        self.features.add_module("relu0", nn.ReLU(inplace=True))
        self.features.add_module("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            self.features.add_module("denseblock%d" % (i + 1),
                DenseBlock(num_layers, num_features, growth_rate, bn_size, drop_rate))
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                out = int(num_features * compression)
                self.features.add_module("transition%d" % (i + 1),
                    Transition(num_features, out))
                num_features = out
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)


def densenet121(**kwargs):
    return DenseNet(32, (6, 12, 24, 16), 64, **kwargs)

def densenet169(**kwargs):
    return DenseNet(32, (6, 12, 32, 32), 64, **kwargs)

def densenet201(**kwargs):
    return DenseNet(32, (6, 12, 48, 32), 64, **kwargs)
```
