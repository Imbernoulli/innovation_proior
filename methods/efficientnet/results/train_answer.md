Image classifiers based on convolutional networks improve by getting bigger, but the usual ways of growing them are unsatisfying. A network has three natural knobs: depth, width, and input resolution. Depth-only scaling, the classic residual-network recipe, makes gradients harder to train and its accuracy return flattens long before one reaches a useful ceiling. Width-only scaling, as in wide residual networks or the width multiplier of mobile networks, is easier to optimize but shallow-and-wide models miss hierarchical features and also saturate. Resolution-only scaling gives finer detail, yet the cost grows quadratically while the gain tapers off. Worse, these dimensions interact: a larger image only helps if the network is deep enough to aggregate that detail and wide enough to represent it. Scaling one knob while freezing the others leaves accuracy on the table, but doing all three by hand is tedious and rarely optimal.

The better approach is to treat depth, width, and resolution as coupled quantities controlled by a single compound coefficient. The method is called EfficientNet. It starts from a small, efficient baseline found by FLOPs-aware architecture search, and then scales that baseline with a simple reusable rule. The baseline block is the mobile inverted bottleneck, or MBConv: it expands a thin representation with a 1x1 convolution, applies a 3x3 or 5x5 depthwise convolution in the expanded space, recalibrates channels with a tiny squeeze-and-excitation unit, and projects back to a thin output with a linear 1x1 convolution. The residual skip connects the thin endpoints when stride and channels match, which keeps memory low. The linear projection is important because a ReLU in a narrow bottleneck can destroy information that cannot be recovered.

The scaling rule itself is the core idea. Let φ be a compound coefficient and let α, β, γ be fixed constants. Then depth is scaled as d = α^φ, width as w = β^φ, and resolution as r = γ^φ. The constraint α β^2 γ^2 ≈ 2 comes from the FLOP cost of a regular convolution, which scales linearly with depth and quadratically with width and resolution, so total cost is roughly d w^2 r^2. With the constraint, φ becomes an approximate compute-doubling knob: φ = 1 is about twice the baseline FLOPs, φ = 2 about four times, and so on. The constants are found once by a small grid search around the small baseline at φ = 1. The reference values are α = 1.2, β = 1.1, and γ = 1.15, whose product satisfies the constraint closely. These constants are then frozen and reused for every larger member of the family. Widths are rounded to multiples of 8, repeats are rounded up, and larger models receive stronger dropout and drop-connect regularization.

The practical EfficientNet family stores the snapped coefficients for each named model, from B0 at width 1.0, depth 1.0, and resolution 224, up to B7 at width 2.0, depth 3.1, and resolution 600. This turns scaling from an expensive per-budget search into a one-time calibration followed by a deterministic rule, while still capturing the coupling between depth, width, and resolution that single-dimension scaling misses.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


BN_MOMENTUM = 0.01
BN_EPS = 1e-3


class Conv2dSamePadding(nn.Conv2d):
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
        Conv2dSamePadding(in_ch, out_ch, kernel_size, stride=stride,
                          groups=groups, bias=False),
        nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM, eps=BN_EPS),
    ]
    if act:
        layers.append(nn.SiLU())
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
    mask = keep + torch.rand([x.shape[0], 1, 1, 1],
                             dtype=x.dtype, device=x.device)
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
        drop_connect_rate=0.2,
        num_classes=num_classes,
    )
    return model, resolution
```
