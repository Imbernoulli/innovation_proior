# Deep residual learning (ResNet), distilled

ResNet makes networks far deeper than was previously trainable by reparameterizing each group
of layers to learn a **residual** added onto a parameter-free **identity shortcut**. Every
building block reduces to one line: `out = F(x) + x`.

## The problem

Depth drives image-recognition accuracy, but you cannot get it by naively stacking layers.
Once variance-preserving initialization and Batch Normalization let deep *plain* nets
converge, a separate **degradation problem** appears: beyond ~20-30 layers, adding layers
*increases training error*. It is

- **not overfitting** — the *training* error itself rises (56-layer plain > 20-layer plain;
  34-layer plain > 18-layer plain);
- **not vanishing/exploding gradients** — with BN the forward responses keep non-zero variance
  and the backward gradients keep healthy norms, both measurable.

It is an **optimization/conditioning** failure. A solution at least as good provably exists: a
deep net can copy a shallow net into its bottom layers and set every extra layer to the
identity, so it should train no worse than the shallow net. SGD cannot find that solution,
because learning an identity (or near-identity) mapping through a stack of conv-BN-ReLU is
hard — and weight decay plus small-init SGD pull weights toward the *zero* (annihilating) map,
not the identity. The capacity is there; the optimizer can't reach it.

## The key idea

**Reparameterize** each group of layers to learn the **residual** instead of the full mapping.
If the desired mapping is `H(x)`, let the layers fit `F(x) = H(x) − x` and recover the output
by adding the input back through a parameter-free identity shortcut:

```
y = F(x, {W_i}) + x
```

Both forms can represent the same functions (they differ by the fixed term `x`), so capacity
is unchanged; only the conditioning changes:

- **Identity becomes free.** If the optimal block mapping is the identity, the solver only has
  to push `F → 0` (drive the conv weights toward zero) — the easiest target there is, and
  exactly where weight decay already pulls. The previously hard case (build identity from
  scratch) is now the easy one.
- **Near-identity is preconditioned.** If the optimum is close to identity, the block learns a
  *small perturbation referenced to x*, not the whole function from nothing.
- **Controlled comparison.** Identity shortcuts add no parameters and no meaningful compute
  (just an element-wise add), so a plain net and its residual counterpart can be matched at
  identical depth, width, parameters, and FLOPs — any improvement is attributable to the
  reformulation, not to extra capacity.
- **Gradient path (a bonus, not the explanation).** At the addition, before the following ReLU,
  `y = F(x) + x` gives the local Jacobian `∂y/∂x = F'(x) + I` (`F' + 1` in scalar shorthand).
  This gives backpropagation a direct additive term through each skip, while later nonlinearities
  still shape the composed Jacobian. Welcome, but the load-bearing claim is conditioning — BN
  already keeps the plain net's gradients healthy.

## Design decisions and why

| Decision | Why this and not the alternative |
|---|---|
| Learn `F = H − x`, recover `H = F + x` | Degradation is an optimization problem; preconditions the solver toward identity (push weights → 0) instead of forcing it to build identity from scratch. |
| **Identity** shortcut, parameter-free | Vs. a Highway-style gate `y = H·T + x·C`: a gate has parameters and can *close* (carry → 0), losing the skip exactly when depth needs it; identity never closes, always passes all of `x`, costs nothing, and keeps the comparison clean. |
| Second ReLU **after** the add, `σ(F + x)` | A ReLU at the end of the residual branch would force `F ≥ 0`, allowing only one-sided corrections; rectify the sum so `F` can be signed. |
| `F` has **≥ 2 layers** | A single-layer residual is `y = (W₁ + I)x`, just a shifted linear layer — no advantage. |
| Dimension match: **option B** (1×1 projection only when channels/stride change; identity elsewhere) | A (zero-pad identity) leaves the padded channels with no residual learning at transitions; C (project every skip) adds many learned shortcuts (thirteen in the 34-layer comparison) and muddies the controlled comparison for marginal gain. B keeps almost every skip free. |
| **Bottleneck** `1×1 reduce → 3×3 → 1×1 restore` (expansion 4) for deep nets | The 3×3 cost is quadratic in channels; running it at the reduced width makes a 3-layer block cost ≈ one 2×(3×3) block, so the same FLOP budget buys far more depth. |
| Keep the bottleneck skip an **identity** | A projection across the bottleneck's two high-dimensional ends (256→256 1×1) roughly **doubles** the block's cost and size; identity makes it free — essential for affordable very-deep nets. |
| GAP head, no fat FC | Global average pooling is parameter-free, a structural regularizer, and translation-robust; VGG's 3-FC head dominates parameters and overfits. |

## The architecture

- **Stem:** 7×7, 64, stride 2 → 3×3 max-pool stride 2.
- **Four stages**, channels 64→128→256→512 (× expansion), following the VGG complexity rule
  (same map size → same #filters; halve map → double filters); downsampling by stride-2 convs at
  the first block of stages 2/3/4.
- **Head:** global average pooling → linear classifier → softmax.
- **BasicBlock** (18/34): two 3×3 convs, `F = W₂ σ(W₁ x)`, expansion 1. Counts: 18 = [2,2,2,2],
  34 = [3,4,6,3].
- **Bottleneck** (50/101/152): 1×1 reduce → 3×3 → 1×1 restore, expansion 4. Counts: 50 = [3,4,6,3],
  101 = [3,4,23,3], 152 = [3,8,36,3]. The 152-layer net (~11.3 GFLOPs) is *lighter* than VGG-19
  (19.6 GFLOPs) despite being 8× deeper.

## Training recipe

BN after every convolution, before activation; rectifier-aware (`kaiming`, fan_out) init; train
from scratch. SGD, mini-batch 256, momentum 0.9, weight decay 1e-4, lr 0.1 divided by 10 on
plateau (up to 60×10⁴ iters). Scale jitter (shorter side ∈ [256, 480]), 224×224 random crop +
flip, per-pixel mean subtraction, color augmentation. No dropout (BN regularizes). Test: 10-crop
or fully-convolutional multi-scale. Very deep CIFAR nets (110+) use a brief 0.01 warm-up until
training error < ~80%, then 0.1.

## Working code

```python
from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1,
            groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element list")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample,
                        self.groups, self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet18():  return ResNet(BasicBlock, [2, 2, 2, 2])
def resnet34():  return ResNet(BasicBlock, [3, 4, 6, 3])
def resnet50():  return ResNet(Bottleneck, [3, 4, 6, 3])
def resnet101(): return ResNet(Bottleneck, [3, 4, 23, 3])
def resnet152(): return ResNet(Bottleneck, [3, 8, 36, 3])
```

Implementation note: the form above follows torchvision's bottleneck stride convention, placing
stage downsampling in the 3×3 (`conv2`). Placing that stride in the first 1×1 is the other common
convention; the residual equation, projection rule, and layer counts are unchanged.
