# ConvNeXt

ConvNeXt is a pure convolutional backbone obtained by modernizing a residual ConvNet with the non-attention design choices that made hierarchical vision Transformers strong: Transformer-style training, a four-stage hierarchy, a patchify stem, separated spatial/channel mixing, an inverted 4x bottleneck, a large depthwise kernel, sparse activations and norms, LayerNorm, separate downsampling, LayerScale, and stochastic depth.

The controlled ResNet-50 / Swin-T path is:

| Step | Top-1 | GFLOPs |
| --- | ---: | ---: |
| ResNet-50, old recipe | 76.13 | 4.09 |
| Modern recipe, no EMA for ablations | 78.82 +- 0.07 | 4.09 |
| Stage counts `(3,4,6,3) -> (3,3,9,3)` | 79.36 +- 0.07 | 4.53 |
| 4x4 stride-4 patchify stem | 79.51 +- 0.18 | 4.42 |
| Depthwise conv alone | 78.28 +- 0.08 | 2.35 |
| Widen base channels `64 -> 96` | 80.50 +- 0.02 | 5.27 |
| Inverted 4x bottleneck | 80.64 +- 0.03 | 4.64 |
| Move depthwise conv before channel MLP | 79.92 +- 0.08 | 4.07 |
| Depthwise kernel 5 | 80.35 +- 0.08 | 4.10 |
| Depthwise kernel 7 | 80.57 +- 0.14 | 4.15 |
| Depthwise kernel 9 | 80.57 +- 0.06 | 4.21 |
| Depthwise kernel 11 | 80.47 +- 0.11 | 4.29 |
| ReLU to GELU | 80.62 +- 0.14 | 4.15 |
| One activation per block | 81.27 +- 0.06 | 4.15 |
| One normalization per block | 81.41 +- 0.09 | 4.15 |
| BatchNorm to LayerNorm | 81.47 +- 0.09 | 4.46 |
| Separate downsampling with boundary LayerNorms | 81.97 +- 0.06 | 4.49 |
| Swin-T reference | 81.30 | 4.50 |

The final block is:

`7x7 depthwise conv -> LayerNorm -> Linear(dim, 4*dim) -> GELU -> Linear(4*dim, dim) -> LayerScale gamma -> DropPath -> residual add`.

The final stage configurations are:

| Variant | Depths | Dims |
| --- | --- | --- |
| ConvNeXt-T | `[3, 3, 9, 3]` | `[96, 192, 384, 768]` |
| ConvNeXt-S | `[3, 3, 27, 3]` | `[96, 192, 384, 768]` |
| ConvNeXt-B | `[3, 3, 27, 3]` | `[128, 256, 512, 1024]` |
| ConvNeXt-L | `[3, 3, 27, 3]` | `[192, 384, 768, 1536]` |
| ConvNeXt-XL | `[3, 3, 27, 3]` | `[256, 512, 1024, 2048]` |

Core PyTorch implementation, faithful to the FAIR reference in `models/convnext.py`:

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

For main ImageNet-1K training after the architecture is fixed, the paper uses AdamW, learning rate `4e-3`, weight decay `0.05`, batch size `4096`, 300 epochs, 20 warmup epochs, cosine decay, RandAugment `(9, 0.5)`, Mixup `0.8`, Cutmix `1.0`, Random Erasing `0.25`, label smoothing `0.1`, LayerScale init `1e-6`, EMA `0.9999`, and stochastic-depth rates `0.1/0.4/0.5/0.5` for T/S/B/L. For ImageNet-22K pretraining, EMA is off and stochastic-depth rates are `0.0/0.0/0.1/0.1/0.2` for T/S/B/L/XL.
