# ResNeXt, distilled

ResNeXt is a residual image-classification network whose building block aggregates `C` transformations of the *same topology* by summation, instead of computing a single transformation. It keeps VGG/ResNet's repeat-the-same-block simplicity and the bottleneck/residual structure, but exposes a new design axis — **cardinality** `C`, the number of aggregated transformations — that raises accuracy at fixed FLOPs and parameter count, more effectively than going deeper or wider.

## The problem

VGG/ResNet stack identical blocks (one clean depth knob, transfers well); Inception gets accuracy-per-FLOP from split-transform-merge but hand-customizes every branch (no transferable knob). The goal: keep the repeat-one-block simplicity *and* the split-transform-merge efficiency, governed by a single turnable parameter, and gain accuracy at *fixed complexity* — rare, and valuable because depth/width give diminishing returns.

## The key idea

The inner product `Σ_{i=1}^{D} w_i x_i` is already split-transform-merge: split `x` into channels, scale each (`w_i x_i`), sum. Generalize the scalar transform into a richer function `T_i` that is itself a network, and aggregate `C` of them:

```
F(x) = Σ_{i=1}^{C} T_i(x),     y = x + Σ_{i=1}^{C} T_i(x)
```

- `C` = **cardinality**, sitting where `D` sat but free to be any value. Width counts simple transforms; cardinality counts complex ones.
- All `T_i` share the **same topology** (VGG philosophy at the sub-block level) → `C` is a single isolated factor, extensible to any value with no per-path design.
- Each `T_i` is a **bottleneck**: `1×1 (256→d) → 3×3 (d→d) → 1×1 (d→256)`; the first 1×1 makes the low-dim embedding.

## Three equivalent forms

1. **(a) Aggregated paths:** `C` bottleneck paths summed, then `+x`.
2. **(b) Concatenation:** `C` paths of `1×1→3×3`, concatenate the `d`-dim outputs into `C·d`, one shared `1×1 (C·d→256)`. Proof of (a)≡(b): `A_1B_1 + A_2B_2 = [A_1,A_2]·[B_1;B_2]` (horizontal concat of last-layer weights, vertical concat of second-last responses).
3. **(c) Grouped convolution:** fuse the `C` input-1×1s into one `1×1 (256→C·d)`; the `3×3` becomes a grouped conv with `C` groups (each group convolves its own `d` channels); then `1×1 (C·d→256)`. Looks like a bottleneck but the wide `3×3` is sparsely connected.

For this homogeneous bottleneck template, all three are strictly equivalent when BN/ReLU are placed consistently; the reformulations are not guaranteed for arbitrary heterogeneous `T_i`, and are nontrivial only for path depth ≥ 3. Implement form (c) — most succinct and fastest. Aggregating jointly-trained transformations is *not* ensembling (ensemble members train independently).

## Capacity / matched complexity

A path costs `256·d + 9·d² + 256·d`; `C` of them give `C·(256·d + 9·d² + 256·d)` parameters (FLOPs proportional). The ResNet bottleneck baseline is `256·64 + 9·64² + 64·256 ≈ 70k`. Trade per-path width `d` (isolated from the fixed 256-channel block I/O) against `C` to hold ~70k:

| cardinality C | 1 | 2 | 4 | 8 | 32 |
|---|---|---|---|---|---|
| bottleneck width d | 64 | 40 | 24 | 14 | 4 |
| grouped-conv width C·d | 64 | 80 | 96 | 112 | 128 |

Default: **C=32, d=4** (`32·(1024+144+1024) ≈ 70k`). Adopt no bottleneck width smaller than `4d` because the preserved-complexity sweep saturates as `d` gets tiny. The decisive test is *training* error — cardinality lowers it, so the gain is stronger representation, not regularization.

## The architecture (ResNeXt-50, 32×4d)

VGG/ResNet template rules (same map → same width; halve map → double width). Stem 7×7/64 stride 2 + 3×3 maxpool; four stages of the block stacked [3,4,6,3]; grouped-3×3 widths 128/256/512/1024, block outputs 256/512/1024/2048; global avg pool → 1000-FC → softmax. ≈25.0M params, 4.2 GFLOPs (vs ResNet-50: 25.5M, 4.1 GFLOPs). Identity shortcuts except dimension-increasing projections (type B); downsample by stride-2 in the 3×3 of each stage's first block; BN after each conv, ReLU after each BN except the block output (ReLU after the add).

## Training recipe

224×224 random crops with scale/aspect-ratio augmentation; SGD, mini-batch 256 on 8 GPUs, momentum 0.9, weight decay 1e-4; LR 0.1 ÷10 three times; He init; BN after every conv. Ablation eval: single 224×224 center crop from a shorter-side-256 image.

## Working code

The canonical release is Torch/Lua with `-bottleneckType resnext_C`; this is a faithful PyTorch transliteration of its form-C block: same `D = floor(planes·baseWidth/64)`, same grouped 3×3 stride placement, same type-B projections, and the same conv/BN/linear-bias initialization.

```python
import torch
import torch.nn as nn
import math


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=32, base_width=4):
        super().__init__()
        D = int(planes * (base_width / 64.0))
        width = D * cardinality                          # C * d
        self.conv1 = conv1x1(inplanes, width)            # 1x1 (in -> C*d)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)   # grouped 3x3
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)  # 1x1 (C*d -> out)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))       # aggregates C same-shape transforms
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity                                  # y = x + sum_i T_i(x)
        return self.relu(out)


class ResNeXt(nn.Module):
    def __init__(self, layers, num_classes=1000, cardinality=32, base_width=4):
        super().__init__()
        self.cardinality = cardinality
        self.base_width = base_width
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResNeXtBottleneck.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        block = ResNeXtBottleneck
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample,
                        self.cardinality, self.base_width)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                cardinality=self.cardinality, base_width=self.base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def resnext50_32x4d():  return ResNeXt([3, 4, 6, 3], cardinality=32, base_width=4)
def resnext101_32x4d(): return ResNeXt([3, 4, 23, 3], cardinality=32, base_width=4)
```
