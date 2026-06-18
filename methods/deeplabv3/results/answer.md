# DeepLabv3

## Method

Use atrous convolution to repurpose an ImageNet-pretrained backbone for dense prediction. For output location **i**,

```text
y[**i**] = sum_**k** x[**i** + r * **k**] * w[**k**].
```

Removing the last stride-2 downsampling and setting subsequent convolutions to rate 2 gives `output_stride = 16`; removing the last two gives `output_stride = 8` with later rates 2 and 4. No extra filter parameters are introduced.

The segmentation head is an improved ASPP module:

- image-level branch: global average pooling -> `1x1` conv with 256 filters + BN + ReLU -> bilinear upsample;
- local branch: `1x1` conv with 256 filters + BN + ReLU;
- three `3x3` atrous branches with 256 filters + BN + ReLU, using rates `(6, 12, 18)` at `output_stride = 16`;
- concatenate all branches -> `1x1` projection with 256 filters + BN + ReLU + dropout -> final `1x1` logits.

At `output_stride = 8`, double the ASPP rates to `(12, 24, 36)`. The large-rate boundary case is why the image-level branch is needed: for a `3x3` atrous filter on an `N x N` feature map, the average normalized count of valid taps is

```text
((N + 2 * max(N - r, 0)) / (3N))^2.
```

When `r >= N`, this becomes `1/9`, so only the center tap is effective and the `3x3` branch has collapsed to a `1x1` operation. DeepLabv3 supplies true global context by pooling the whole feature map instead of relying on an extreme dilation rate. The model drops the DenseCRF post-processing used by earlier DeepLab versions.

## Code

```python
import torch
from torch import nn
from torch.nn import functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for module in self:
            x = module(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super().__init__()
        branches = [
            ASPPPooling(in_channels, out_channels),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ),
        ]
        branches.extend(
            ASPPConv(in_channels, out_channels, rate)
            for rate in atrous_rates
        )
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(len(branches) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class DeepLabV3Head(nn.Sequential):
    def __init__(self, in_channels, num_classes, atrous_rates=(6, 12, 18)):
        super().__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )


class DeepLabV3(nn.Module):
    def __init__(self, backbone, num_classes, in_channels=2048, atrous_rates=(6, 12, 18)):
        super().__init__()
        self.backbone = backbone
        self.classifier = DeepLabV3Head(in_channels, num_classes, atrous_rates)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)["out"]
        logits = self.classifier(features)
        return F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=True)
```

Use `(6, 12, 18)` for a backbone configured to `output_stride = 16`; use `(12, 24, 36)` for `output_stride = 8`. A paper-faithful ResNet backbone removes the relevant late strides and replaces them with dilation rather than adding decoder layers or CRF post-processing.
