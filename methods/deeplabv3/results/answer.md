# DeepLabv3

## Problem

Semantic segmentation with an ImageNet-pretrained DCNN faces two obstacles: (1) repeated pooling/striding reduces the feature map by ~32×, destroying the spatial detail dense prediction needs; (2) objects appear at many scales, which a single fixed receptive field cannot handle. DeepLabv3 solves both with atrous convolution: it extracts dense features at a chosen resolution and probes them at multiple fields-of-view, in one end-to-end network with no DenseCRF post-processing.

## Key idea

**Atrous convolution.** y[**i**] = Σ_**k** x[**i** + r·**k**] · w[**k**]. The rate r inserts r−1 zeros between filter taps, enlarging the field-of-view with no extra parameters and no resolution loss. Define **output_stride** = input resolution / feature-map resolution. To make a 32× backbone produce denser features, remove the stride from a late downsampling layer and set the subsequent convs to rate r (r=2 gives output_stride 16, then r=4 gives output_stride 8), preserving each filter's field-of-view.

**Two layouts for multi-scale context.**
- *Cascade*: duplicate ResNet's last block (block4 → block5/6/7) but keep resolution fixed via atrous rates instead of striding (consecutive striding decimates detail). Optional **multi-grid**: the three 3×3 convs in a block get unit rates (r1,r2,r3), with actual rate = block_rate × unit_rate (e.g. output_stride 16, Multi_Grid (1,2,4) → rates (2,4,8)).
- *Parallel = ASPP*: parallel atrous convolutions at different rates probe one feature map at multiple fields-of-view.

**ASPP degeneration + image-level features.** As the atrous rate approaches the feature-map size, fewer filter taps land on valid (non-padding) feature positions; in the limit a 3×3 atrous conv degenerates to a 1×1 conv (only the center tap is effective), capturing no global context. Fix: add an **image-level branch** — global average pool → 1×1 conv (256) + BN → bilinear upsample.

**Final ASPP (output_stride 16):** five parallel branches, each 256 channels + BN + ReLU: (a) one 1×1 conv; (b) three 3×3 atrous convs at rates (6, 12, 18); (c) the image-level branch. Concatenate → 1×1 conv (256) + BN + ReLU (+ dropout) → final 1×1 conv → logits. Rates double to (12, 24, 36) at output_stride 8. Batch norm inside ASPP is trained (new vs DeepLabv2). The coarse logits are bilinearly upsampled to the input resolution.

## Code

```python
import torch
from torch import nn
from torch.nn import functional as F

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super().__init__()
        modules = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU())]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout(0.5))

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        return self.project(torch.cat(res, dim=1))

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes, atrous_rates=(6, 12, 18)):
        super().__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )

class DeepLabV3(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone       # ResNet with dilated late stages (output_stride 16 or 8)
        self.classifier = classifier   # DeepLabHead(2048, num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)["out"]
        x = self.classifier(features)
        return F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
```

The backbone is a ResNet-50/101 whose last stage(s) drop the downsampling stride and apply atrous convolution to reach output_stride 16 (rates (6,12,18)) or 8 (rates (12,24,36)). Trained with SGD and a poly learning-rate schedule (lr × (1 − iter/max_iter)^0.9), large crop (~513), and fine-tuned batch norm; evaluated by mean IoU on PASCAL VOC 2012 / Cityscapes.
