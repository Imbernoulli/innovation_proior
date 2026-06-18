# MobileNet v1

## Core formula

For an input `D_F x D_F x M`, output depth `N`, and `D_K x D_K` kernel, a standard convolution costs

```text
D_K^2 M N D_F^2.
```

A depthwise separable block replaces it with:

```text
depthwise: D_K^2 M D_F^2
pointwise: M N D_F^2
total:     D_K^2 M D_F^2 + M N D_F^2
ratio:     1/N + 1/D_K^2
```

With `3 x 3` kernels and typical `N >= 64`, this is about `8x` to `9x` less computation. Width and resolution multipliers give the combined cost

```text
D_K^2 alpha M (rho D_F)^2 + alpha M alpha N (rho D_F)^2.
```

The depthwise term is linear in `alpha`; the dominant pointwise term is quadratic. Resolution `rho` changes compute by `rho^2` and does not change parameter count.

## Architecture

The network uses one full `3 x 3` stride-2 stem, then 13 depthwise-separable blocks:

```text
(64,1), (128,2), (128,1), (256,2), (256,1),
(512,2), (512,1), (512,1), (512,1), (512,1), (512,1),
(1024,2), (1024,1)
```

Each tuple is `(output_channels, stride)`, and the stride is applied in the depthwise convolution. The final `1024` block is stride 1, matching the canonical implementation and the retained `7 x 7` feature map. The head is average pooling to `1 x 1`, dropout, and a `1 x 1` logits layer; softmax is a prediction function over logits.

## Faithful PyTorch transcription

```python
import torch
import torch.nn as nn


def conv_bn_relu6(in_ch, out_ch, kernel, stride, padding, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = conv_bn_relu6(
            in_ch, in_ch, kernel=3, stride=stride, padding=1, groups=in_ch
        )
        self.pointwise = conv_bn_relu6(
            in_ch, out_ch, kernel=1, stride=1, padding=0
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MobileNetV1(nn.Module):
    cfg = [
        (64, 1), (128, 2), (128, 1), (256, 2), (256, 1),
        (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
        (1024, 2), (1024, 1),
    ]

    def __init__(self, num_classes=1000, width_mult=1.0, min_depth=8,
                 dropout_keep_prob=0.999):
        super().__init__()

        def depth(ch):
            return max(int(ch * width_mult), min_depth)

        in_ch = depth(32)
        self.stem = conv_bn_relu6(3, in_ch, kernel=3, stride=2, padding=1)

        blocks = []
        for out_ch, stride in self.cfg:
            out_ch = depth(out_ch)
            blocks.append(DepthwiseSeparableBlock(in_ch, out_ch, stride))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=1.0 - dropout_keep_prob)
        self.logits = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.logits(x)
        return torch.flatten(x, 1)
```

The official TF-Slim code represents the same block as `slim.separable_conv2d(net, None, [3, 3], depth_multiplier=1, stride=s)` for the depthwise step, followed by `slim.conv2d(net, depth, [1, 1], stride=1)` for pointwise mixing. Its width multiplier is named `depth_multiplier` in code and applies `depth = max(int(d * depth_multiplier), min_depth)`. The arg scope uses batch normalization, `tf.nn.relu6`, and no depthwise weight regularizer by default.
