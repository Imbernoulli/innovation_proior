Semantic segmentation asks for a class label at every pixel, so it needs both the object-recognition power of a classification network and a spatial resolution that classification networks deliberately discard. The standard ImageNet-pretrained backbone is attractive because its late layers already encode semantics, but it is built around repeated stride-2 operations that shrink the feature map to one thirty-second of the input size. If we simply attach a classifier to that coarse grid and upsample, we force the model to reconstruct boundaries and small objects from information that has already been thrown away.

Existing ideas each leave a gap. Fully convolutional networks keep whole-image computation efficient but their score map is too coarse. Encoder-decoder designs such as U-Net add a learned decoder to recover detail, which increases complexity and still has to reconstruct fine structure after aggressive downsampling. Earlier atrous segmentation systems do compute denser feature maps, but they rely on a separate DenseCRF post-process for boundaries and their pyramid context module does not train batch-normalization layers inside the added head. Running an image pyramid handles scale but multiplies compute, while cascaded context modules that keep striding collapse the feature map again. What is needed is a single forward pass that keeps the pretrained backbone useful, preserves spatial resolution, and aggregates multi-scale context without extra post-processing.

The method is DeepLabv3. Its first move is to repurpose the pretrained backbone with atrous convolution. Atrous convolution inserts holes between filter taps: for output location i, it computes y[i] = sum_k x[i + r * k] * w[k], where r is the atrous rate. When a stride-2 downsampling is removed, the following convolutions can use rate 2 to keep roughly the same input receptive field while producing a denser grid. Removing the last downsampling gives output_stride 16; removing the last two gives output_stride 8, with later rates 2 and 4. The parameter count stays the same and the pretrained weights remain useful.

The second move is an improved Atrous Spatial Pyramid Pooling module. A single field of view cannot suit every object scale, so the head uses several parallel branches over the same dense feature map: a 1x1 local branch, three 3x3 atrous branches with different rates, and an image-level global branch. The image-level branch is important because very large atrous rates degenerate: on an N x N feature map, a 3x3 atrous filter has average normalized valid taps equal to ((N + 2 * max(N - r, 0)) / (3N))^2, which becomes 1/9 once r >= N. In other words, the outer taps fall off the map and the branch collapses to a 1x1 operation. Global context is therefore supplied explicitly by adaptive average pooling, a 1x1 convolution, and bilinear upsampling, rather than by pushing the atrous rate to extremes.

At output_stride 16, the ASPP rates are (6, 12, 18); at output_stride 8 they are doubled to (12, 24, 36) to match the denser grid. Each branch produces 256 channels with batch normalization and ReLU. The branches are concatenated, projected to 256 channels by another 1x1 convolution with batch normalization, ReLU, and dropout, and finally mapped to class logits by a 1x1 convolution. The logits are bilinearly upsampled to the input resolution during inference. There is no DenseCRF post-processing, no separate decoder, and no extra 3x3 refinement after ASPP in the core formulation.

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

Use atrous rates (6, 12, 18) with a backbone configured to output_stride 16, and (12, 24, 36) with output_stride 8. The ResNet backbone should remove the relevant late strides and replace them with dilation, keeping the pretrained weights and avoiding additional decoder layers or CRF post-processing.
