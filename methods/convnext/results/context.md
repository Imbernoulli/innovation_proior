# Context

## Research question

By 2021, the usual backbone comparison in visual recognition has become hard to interpret. A hierarchical vision Transformer can beat a conventional residual ConvNet on classification, detection, and segmentation, but it differs from that ConvNet along many axes at once: optimizer, schedule length, augmentation, regularization, stage layout, input stem, downsampling rule, normalization, activation placement, channel expansion, and spatial mixing primitive.

The question is therefore not "which system wins?" but "which design differences account for the gap?" A useful reconstruction has to separate attention from the rest of the Transformer-era bundle. It should keep compute in the same regime, change one design family at a time, and ask whether the non-attention choices already explain most of the observed advantage.

## Background

Convolutional networks bring locality, translation equivariance, and shared sliding-window computation. These properties are especially useful when an image backbone must support dense prediction at high resolution, because the same spatial operator can be reused across positions and resolutions. Through the 2010s, the main ConvNet line refined how to allocate that computation: VGG favored repeated small kernels, ResNet made very deep stacks trainable with residual mappings, ResNeXt exposed grouped-convolution cardinality, MobileNet and Xception separated spatial and channel computation with depthwise separable convolutions, MobileNetV2 used inverted residuals, and EfficientNet/RegNet studied scalable design spaces.

Vision Transformers changed the comparison. ViT treats non-overlapping image patches as tokens and applies a standard Transformer encoder: LayerNorm before multi-head self-attention, residual add, LayerNorm before a two-layer MLP, residual add, with GELU in the MLP and a hidden dimension commonly four times the token dimension. This greatly reduces image-specific priors, but global self-attention has quadratic cost in token count and becomes awkward for dense high-resolution tasks.

Hierarchical vision Transformers respond by reintroducing image-like structure: multiple stages with decreasing resolution and increasing width, local windows instead of global attention, and separate patch-merging downsampling between stages. These choices make the model usable as a generic vision backbone, but they also make the comparison with a plain residual ConvNet confounded.

## Baselines

**Residual ConvNet.** A standard residual backbone has a 7x7 stride-2 convolution plus max-pool stem, four stages with bottleneck blocks, BatchNorm and ReLU around each convolution, residual additions, and in-block stride-2 downsampling at stage boundaries. Its reference block counts in the 50-layer regime are `(3, 4, 6, 3)`.

**Grouped and depthwise convolution.** ResNeXt shows that increasing cardinality with grouped convolution can improve the FLOPs/accuracy trade-off, provided width is adjusted. Depthwise convolution is the extreme grouped case, with `groups == channels`; it filters each channel spatially, while a following 1x1 convolution mixes channels.

**Inverted residuals.** MobileNetV2 uses a narrow-to-wide-to-narrow residual block: a 1x1 expansion, a lightweight depthwise spatial convolution, and a linear 1x1 projection back to the narrow dimension. The skip connects the narrow endpoints, and avoiding nonlinearities at the narrow projection is part of the design.

**Vision Transformer family.** ViT supplies the patch embedding, pre-normalized attention/MLP block, GELU, and 4x MLP expansion. A hierarchical windowed Transformer supplies the four-stage layout, local windows, shifted-window communication, patch merging, widths that double between stages, and a small-patch stem.

## Evaluation settings

The controlled architecture path should use ImageNet-1K top-1 accuracy at fixed training settings, with the small regime near ResNet-50 / hierarchical-Transformer-tiny compute (about 4.5 GFLOPs) and a larger regime near ResNet-200 / hierarchical-Transformer-base compute (about 15 GFLOPs). The starting ConvNet must first be retrained with the modern recipe used by the Transformer-era backbones: AdamW, 300 epochs, 20-epoch warmup, cosine decay, batch size 4096, weight decay 0.05, Mixup, Cutmix, RandAugment, Random Erasing, label smoothing, and stochastic depth. For the stepwise modernization experiments, EMA should be disabled when BatchNorm-containing variants are still being compared.

After the architecture is fixed, system-level comparisons should include ImageNet-1K training, ImageNet-22K pretraining followed by ImageNet-1K fine-tuning, COCO detection/instance segmentation with Mask R-CNN or Cascade Mask R-CNN, and ADE20K semantic segmentation with UperNet. Useful metrics are top-1 accuracy, FLOPs, parameters, throughput/FPS, box AP, mask AP, and mIoU.

## Code framework

The scaffold is a generic four-stage convolutional backbone. It deliberately leaves the stem, downsampling rule, residual branch, normalization, activation placement, channel expansion, and final pooling/norm choices open.

```python
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.drop_path = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.block_body(x)
        return residual + self.drop_path(x)

    def block_body(self, x):
        raise NotImplementedError

class Backbone(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, depths=None, dims=None):
        super().__init__()
        self.depths = depths
        self.dims = dims
        self.stem = self.make_stem(in_chans, dims[0])
        self.downsample_layers = nn.ModuleList(
            [self.make_downsample(dims[i], dims[i + 1]) for i in range(3)]
        )
        self.stages = nn.ModuleList(
            [nn.Sequential(*[Block(dims[i]) for _ in range(depths[i])]) for i in range(4)]
        )
        self.norm = self.make_final_norm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def make_stem(self, in_chans, out_dim):
        raise NotImplementedError

    def make_downsample(self, in_dim, out_dim):
        raise NotImplementedError

    def make_final_norm(self, dim):
        raise NotImplementedError

    def forward_features(self, x):
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsample_layers[i - 1](x)
            x = stage(x)
        raise NotImplementedError

    def forward(self, x):
        return self.head(self.forward_features(x))
```
