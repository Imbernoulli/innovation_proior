## Research Question

The problem is to build a basic convolutional image-recognition architecture for a very small compute envelope: roughly tens to hundreds of MFLOPs, the range where a phone, drone, or small robot can plausibly run a model repeatedly. The goal is not to prune, quantize, or distill a large network after training. The goal is to choose the building block itself so the network is accurate at the target budget from the start.

The hard constraint is not only parameter count. The model must be compared by multiply-adds at matched budgets and by actual latency on an ARM-class mobile CPU, because a layer that is cheap in FLOPs can still be slow if it has poor memory access behavior. The target task is standard ImageNet classification, with object detection transfer as a check that the representation is not overfit to classification.

## Background

Residual networks give a reliable template: a shortcut plus a bottleneck residual branch. In the bottleneck version, the branch is `1x1 reduce -> 3x3 spatial -> 1x1 expand`, so if the input/output width is `c`, the bottleneck width is `m`, and the feature map is `h x w`, the dense block costs `hw(2cm + 9m^2)`. The two `1x1` layers are the `2cm` channel-mixing cost; the `3x3` layer is the `9m^2` spatial cost.

Grouped convolution is also available. It divides channels into groups and performs independent convolutions inside each group, cutting the cost of that layer by about the number of groups. ResNeXt uses this idea on the `3x3` bottleneck layer, making a block cost `hw(2cm + 9m^2/g)`. That improves large residual networks, but it leaves the two `1x1` pointwise layers dense.

Depthwise separable convolution takes the opposite extreme for spatial filtering: one `3x3` filter per channel, followed by a dense `1x1` pointwise convolution for channel mixing. Xception and MobileNet make this pattern practical for efficient networks. The important warning is that depthwise convolution is not automatically fast on mobile hardware because it has a weak compute-to-memory-access ratio. Xception also supplies a useful implementation detail: avoid putting a ReLU immediately after the depthwise operation.

The diagnostic pressure in the tiny-network regime is that once the spatial term has been reduced, dense pointwise convolutions dominate the remaining work. In the grouped-`3x3` residual setting with cardinality 32, the pointwise layers account for 93.4% of the unit's multiply-adds. Under a fixed MFLOP budget that dominance forces thin feature maps, and thin feature maps are a bad bargain for a small network that already has too little representational capacity.

## Baselines

**Residual bottleneck.** Reliable optimization and an identity shortcut, but the unit cost `hw(2cm + 9m^2)` is too high when the budget is only tens of MFLOPs. Both channel mixing and spatial filtering remain dense.

**Cardinality-style grouped residual block.** Grouping the `3x3` term gives `hw(2cm + 9m^2/g)`, but the unchanged `2cm` pointwise term becomes the dominant cost at small scale. It saves compute in the wrong place for the mobile budget.

**Depthwise-separable lightweight block.** Depthwise spatial filtering is extremely cheap, and MobileNet shows this can be a strong lightweight baseline. Its dense pointwise convolutions still do all cross-channel mixing, so at very small widths they remain the main compute bottleneck.

**Post-training acceleration.** Pruning, low-rank factorization, quantization, and distillation can shrink an existing model, but they do not answer which basic unit should be used before compression.

## Evaluation Settings

The primary classification setting is ImageNet-2012 with single-crop top-1 error on the validation set: resize the image to 256 on the short side and evaluate a centered `224 x 224` crop. Comparisons should hold FLOPs roughly fixed, especially around 140M, 40M, and 13M multiply-adds.

The training recipe can follow the grouped-residual recipe but must be adapted for very small networks, which tend to underfit rather than overfit. That means lighter weight decay (`4e-5` rather than `1e-4`), a linearly decayed learning rate from `0.5` to `0`, less aggressive scale augmentation, batch size 1024 on 4 GPUs, and about `3e5` iterations.

The deployment check should include actual single-thread latency on an ARM-based mobile processor, not only theoretical FLOPs. A transfer check on MS COCO detection can test whether the representation learned by the small classifier remains useful beyond ImageNet labels.

## Code Framework

The scaffold is a residual image-classification harness. The open slot is the unit repeated in the three main stages: it must decide how to spend channel-mixing and spatial-filtering compute, how to handle stride-2 downsampling, and how to preserve information flow when channels are split.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_ch, out_ch, groups=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, groups=groups, bias=False)


def depthwise3x3(channels, stride=1):
    return nn.Conv2d(channels, channels, kernel_size=3, stride=stride,
                     padding=1, groups=channels, bias=False)


class Unit(nn.Module):
    def __init__(self, in_ch, out_ch, *, groups, stride):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class SmallBudgetNet(nn.Module):
    def __init__(self, unit_cls, stage_channels, groups=3, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, stage_channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stage_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.stage2 = self._make_stage(unit_cls, stage_channels[0], stage_channels[1], 4, groups)
        self.stage3 = self._make_stage(unit_cls, stage_channels[1], stage_channels[2], 8, groups)
        self.stage4 = self._make_stage(unit_cls, stage_channels[2], stage_channels[3], 4, groups)
        self.fc = nn.Linear(stage_channels[3], num_classes)

    def _make_stage(self, unit_cls, in_ch, out_ch, repeats, groups):
        layers = [unit_cls(in_ch, out_ch, groups=groups, stride=2)]
        for _ in range(repeats - 1):
            layers.append(unit_cls(out_ch, out_ch, groups=groups, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.stem(x))
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)
```
