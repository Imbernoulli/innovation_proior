By 2021, hierarchical vision Transformers had overtaken residual ConvNets on ImageNet-1K and downstream dense tasks, and the obvious explanation was attention. But the comparison is confounded: the two families also differ in training recipe, stem design, stage allocation, normalization, activation placement, block shape, and downsampling strategy. To find out how much of the gap is genuinely due to attention, I need to hold the modern training recipe fixed and transplant the non-attention choices from the Transformer era back into a pure ConvNet, changing one family of design decisions at a time.

The result of that step-by-step modernization is surprising: the residual ConvNet closes the controlled gap to Swin-T without ever adding self-attention. The dominant gains come not from attention, but from a bundle of macro and micro design changes that had been wrapped around attention. I call the resulting architecture ConvNeXt.

ConvNeXt is a pure convolutional backbone that deliberately copies the useful non-attention parts of the Transformer-era vision design package. It keeps a four-stage hierarchy, but uses Transformer-style stage ratios and channel dimensions: for the tiny variant the stage depths are [3, 3, 9, 3] and the channel dimensions are [96, 192, 384, 768]. The stem is a single 4x4 stride-4 convolution rather than the traditional 7x7 stride-2 plus max-pool pair, producing the same initial 4x downsampling with a simpler non-overlapping patchify projection. Between stages the downsampling is performed by a standalone 2x2 stride-2 convolution, each time preceded by LayerNorm to keep training stable.

The block itself separates spatial and channel mixing, just like a Transformer block separates self-attention from the MLP. Spatial mixing is done by a 7x7 depthwise convolution, placed at the beginning of the block before the channel expansion. Channel mixing uses two 1x1 pointwise convolutions implemented as linear layers after permuting to channels-last layout: first a 4x expansion, then a GELU, then a projection back to the original dimension. The block applies only one normalization and one activation per block, matching the Transformer block's sparsity rather than the ConvNet habit of normalizing and activating after every convolution. LayerNorm replaces BatchNorm in this remodeled block, and the residual branch is scaled by a per-channel learnable LayerScale parameter initialized near zero, followed by stochastic depth.

The implementation below follows the FAIR reference faithfully. A custom LayerNorm supports both channels-first and channels-last layouts, because the stem and downsampling layers keep channels-first tensors while the block body works channels-last for efficiency with linear layers.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        ) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., layer_scale_init_value=1e-6,
                 head_init_scale=1.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def convnext_tiny(**kwargs):
    return ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)


def convnext_small(**kwargs):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)


def convnext_base(**kwargs):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)


def convnext_large(**kwargs):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)


def convnext_xlarge(**kwargs):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
```

For ImageNet-1K training, the model uses AdamW with learning rate 4e-3, weight decay 0.05, batch size 4096, 300 epochs with a 20-epoch linear warmup and cosine decay, RandAugment, Mixup, Cutmix, Random Erasing, label smoothing 0.1, LayerScale initialization 1e-6, EMA 0.9999, and stochastic-depth rates 0.1/0.4/0.5/0.5 for the T/S/B/L variants. ConvNeXt thus demonstrates that, once training and architecture are modernized together, a ConvNet can match or exceed a hierarchical vision Transformer of the same compute regime, without needing self-attention at all.
