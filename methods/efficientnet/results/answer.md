# EfficientNet: compound model scaling

## Problem

Given a baseline ConvNet, scale it to larger resource budgets without re-searching every stage and without pushing only one saturating dimension. The scalable quantities are depth, width, and input resolution.

## Method

Use one compound coefficient `phi` and three constants:

```text
depth      d = alpha^phi
width      w = beta^phi
resolution r = gamma^phi
subject to alpha * beta^2 * gamma^2 ~= 2
```

The FLOPs of a regular convolution scale linearly with depth and quadratically with width and resolution: `FLOPs ~= d * w^2 * r^2`. Therefore the one-step compute multiplier is `(alpha * beta^2 * gamma^2)^phi`; constraining `alpha * beta^2 * gamma^2 ~= 2` makes `phi` an approximate compute-doubling knob.

For the searched baseline, a small grid search at `phi = 1` gives:

```text
alpha = 1.2
beta  = 1.1
gamma = 1.15
1.2 * 1.1^2 * 1.15^2 = 1.9203 ~= 2
```

The ratios are found once on the small baseline, then reused for larger models. The practical family stores rounded coefficients:

| model | width | depth | resolution | dropout |
|---|---:|---:|---:|---:|
| B0 | 1.0 | 1.0 | 224 | 0.2 |
| B1 | 1.0 | 1.1 | 240 | 0.2 |
| B2 | 1.1 | 1.2 | 260 | 0.3 |
| B3 | 1.2 | 1.4 | 300 | 0.3 |
| B4 | 1.4 | 1.8 | 380 | 0.4 |
| B5 | 1.6 | 2.2 | 456 | 0.4 |
| B6 | 1.8 | 2.6 | 528 | 0.5 |
| B7 | 2.0 | 3.1 | 600 | 0.5 |

## Baseline Architecture

The baseline is found by a FLOPs-aware NAS search using the MnasNet search space and objective `ACC(m) * (FLOPS(m) / T)^w`, with `w = -0.07` and target `T` near 400M FLOPs. Its block is MBConv: optional `1x1` expansion, depthwise convolution, squeeze-and-excitation, and a linear `1x1` projection, with a residual skip only when stride is 1 and channels match.

| stage | op | input resolution | output channels | repeats |
|---|---|---:|---:|---:|
| stem | Conv3x3 s2 | 224 | 32 | 1 |
| 1 | MBConv1 k3 | 112 | 16 | 1 |
| 2 | MBConv6 k3 | 112 | 24 | 2 |
| 3 | MBConv6 k5 | 56 | 40 | 2 |
| 4 | MBConv6 k3 | 28 | 80 | 3 |
| 5 | MBConv6 k5 | 14 | 112 | 3 |
| 6 | MBConv6 k5 | 14 | 192 | 4 |
| 7 | MBConv6 k3 | 7 | 320 | 1 |
| head | Conv1x1, pool, FC | 7 | 1280 | 1 |

All MBConv stages use squeeze-and-excitation ratio `0.25`; SE reduction is computed from the block input filters.

## Reference-Faithful Code Artifact

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


BN_MOMENTUM = 0.01  # PyTorch convention matching TensorFlow momentum 0.99
BN_EPS = 1e-3


class Conv2dSamePadding(nn.Conv2d):
    """TensorFlow-style SAME padding, including stride-2 odd-size cases."""

    def forward(self, x):
        ih, iw = x.shape[-2:]
        kh, kw = self.weight.shape[-2:]
        sh, sw = self.stride
        dh, dw = self.dilation
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
        if pad_h or pad_w:
            x = F.pad(x, [
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2,
            ])
        return F.conv2d(
            x, self.weight, self.bias, self.stride,
            (0, 0), self.dilation, self.groups
        )


def conv_bn_act(in_ch, out_ch, kernel_size, stride=1, groups=1, act=True):
    layers = [
        Conv2dSamePadding(
            in_ch, out_ch, kernel_size, stride=stride,
            groups=groups, bias=False
        ),
        nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM, eps=BN_EPS),
    ]
    if act:
        layers.append(nn.SiLU())  # Swish-1: x * sigmoid(x)
    return nn.Sequential(*layers)


def round_filters(channels, width_coeff, divisor=8, min_depth=None):
    if not width_coeff:
        return channels
    channels *= width_coeff
    min_depth = min_depth or divisor
    new_c = max(min_depth, int(channels + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * channels:
        new_c += divisor
    return int(new_c)


def round_repeats(repeats, depth_coeff):
    if not depth_coeff:
        return repeats
    return int(math.ceil(depth_coeff * repeats))


def drop_connect(x, drop_rate, training):
    if not training or drop_rate <= 0.0:
        return x
    keep = 1.0 - drop_rate
    mask = keep + torch.rand(
        [x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device
    )
    mask = torch.floor(mask)
    return x / keep * mask


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride,
                 expand_ratio=6, se_ratio=0.25):
        super().__init__()
        self.use_skip = stride == 1 and in_ch == out_ch
        mid = in_ch * expand_ratio

        self.expand = (
            conv_bn_act(in_ch, mid, 1)
            if expand_ratio != 1 else nn.Identity()
        )
        self.depthwise = conv_bn_act(
            mid, mid, kernel_size, stride=stride, groups=mid
        )

        se_ch = max(1, int(in_ch * se_ratio))
        self.se_reduce = nn.Conv2d(mid, se_ch, 1)
        self.se_expand = nn.Conv2d(se_ch, mid, 1)

        self.project = conv_bn_act(mid, out_ch, 1, act=False)

    def forward(self, x, drop_rate=0.0):
        h = self.expand(x)
        h = self.depthwise(h)
        s = F.adaptive_avg_pool2d(h, 1)
        s = self.se_expand(F.silu(self.se_reduce(s)))
        h = torch.sigmoid(s) * h
        h = self.project(h)
        if self.use_skip:
            h = drop_connect(h, drop_rate, self.training)
            h = h + x
        return h


# (expand, kernel, stride, input, output, repeats), matching the official strings:
# r1_k3_s11_e1_i32_o16_se0.25, ..., r1_k3_s11_e6_i192_o320_se0.25
BASE_STAGES = [
    (1, 3, 1, 32, 16, 1),
    (6, 3, 2, 16, 24, 2),
    (6, 5, 2, 24, 40, 2),
    (6, 3, 2, 40, 80, 3),
    (6, 5, 1, 80, 112, 3),
    (6, 5, 2, 112, 192, 4),
    (6, 3, 1, 192, 320, 1),
]


class EfficientNet(nn.Module):
    def __init__(self, width_coeff=1.0, depth_coeff=1.0,
                 dropout=0.2, drop_connect_rate=0.2, num_classes=1000):
        super().__init__()
        self.drop_connect_rate = drop_connect_rate

        stem_ch = round_filters(32, width_coeff)
        self.stem = conv_bn_act(3, stem_ch, 3, stride=2)

        blocks = []
        in_ch = stem_ch
        for expand, kernel, stride, _base_in, base_out, repeats in BASE_STAGES:
            out_ch = round_filters(base_out, width_coeff)
            for idx in range(round_repeats(repeats, depth_coeff)):
                blocks.append(MBConv(
                    in_ch, out_ch, kernel,
                    stride if idx == 0 else 1,
                    expand_ratio=expand,
                    se_ratio=0.25,
                ))
                in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        head_ch = round_filters(1280, width_coeff)
        self.head = conv_bn_act(in_ch, head_ch, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(head_ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        total = len(self.blocks)
        for idx, block in enumerate(self.blocks):
            drop_rate = self.drop_connect_rate * float(idx) / total
            x = block(x, drop_rate)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


CONFIGS = {
    "b0": (1.0, 1.0, 224, 0.2),
    "b1": (1.0, 1.1, 240, 0.2),
    "b2": (1.1, 1.2, 260, 0.3),
    "b3": (1.2, 1.4, 300, 0.3),
    "b4": (1.4, 1.8, 380, 0.4),
    "b5": (1.6, 2.2, 456, 0.4),
    "b6": (1.8, 2.6, 528, 0.5),
    "b7": (2.0, 3.1, 600, 0.5),
}


def build(name="b0", num_classes=1000):
    width, depth, resolution, dropout = CONFIGS[name.lower()]
    model = EfficientNet(
        width_coeff=width,
        depth_coeff=depth,
        dropout=dropout,
        drop_connect_rate=0.2,  # official survival_prob = 0.8
        num_classes=num_classes,
    )
    return model, resolution
```
