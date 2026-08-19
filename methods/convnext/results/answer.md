# ConvNeXt

ConvNeXt is a pure convolutional backbone proposed by modernizing a residual ConvNet with the non-attention design choices that made hierarchical vision Transformers strong: Transformer-style training, a four-stage hierarchy, a patchify stem, separated spatial/channel mixing, an inverted 4x bottleneck, a large depthwise kernel, sparse activations and norms, LayerNorm, separate downsampling, LayerScale, and stochastic depth.

The controlled ResNet-50 / Swin-T validation path holds compute in the same ~4.5 GFLOPs regime throughout and changes one design axis per step, each matched against its immediate predecessor's training protocol so a change can only be credited to the axis that moved:

1. Retrain the plain ResNet-50 with the modernized recipe (AdamW, 300 epochs, RandAugment, Mixup, Cutmix, Random Erasing, label smoothing, stochastic depth, no EMA while BatchNorm variants remain in the path) — this becomes the baseline every later step is measured against, so an architectural gain can't be confused with what the recipe alone buys.
2. Stage counts `(3,4,6,3) -> (3,3,9,3)`, matching the Transformer-era 1:1:3:1 depth ratio.
3. Replace the 7x7-stride-2-conv-plus-maxpool stem with a single 4x4 stride-4 patchify convolution (the two stems are verified by hand to give stage 1 the same 56x56 grid, so only the manner of downsampling changes).
4. Make the 3x3 convolution depthwise, then widen the base channel count `64 -> 96` to spend the compute a depthwise conv frees back as width, following ResNeXt's grouped-convolution precedent.
5. Invert the block to a narrow-wide-narrow shape with a 4x channel expansion, matching MobileNetV2's inverted residual to the Transformer MLP's expansion ratio.
6. Move the depthwise convolution ahead of the channel expansion, so the spatial mixer acts on the narrow tensor rather than the expanded one.
7. Sweep the depthwise kernel size (3/5/7/9/11) at matched compute. A MAC count at this point shows channel mixing outweighs spatial mixing roughly 16x at this width, so the predicted FLOP curve across the sweep is close to flat and accuracy is expected to show diminishing returns that saturate near the local-window scale (7x7) that the Swin-T competitor uses at this stage.
8. Replace ReLU with GELU, matching the Transformer MLP's activation.
9. Drop every block activation except the one between the two channel-mixing layers, testing whether the conventional ConvNet block was over-saturated with nonlinearity rather than under-nonlinear.
10. Drop every block normalization except one, placed before the first 1x1 layer.
11. Replace BatchNorm with LayerNorm inside this remodeled (depthwise-mixed, sparsely-activated, singly-normalized) block.
12. Replace in-block downsampling with a standalone 2x2 stride-2 convolution between stages, with a LayerNorm added at every resolution-change boundary (after the stem, before each downsampling conv, after the final pool) to compensate for the normalization that in-block downsampling used to co-locate with every resolution change — the boundary most likely to destabilize training if left unnormalized.

The decision rule carried through the whole chain: a change is kept if it holds or improves accuracy at matched compute; a change that costs accuracy without a compensating structural reason (like the width increase compensating for depthwise) is dropped or reworked. The chain is run once at the small ResNet-50/Swin-T regime and once at the larger ResNet-200/Swin-B regime (~15 GFLOPs) against the corresponding Swin references, so a design choice earns a place in the final architecture only if the sign of its effect agrees at both scales, not just one.

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

For main ImageNet-1K training after the architecture is fixed, I use AdamW, learning rate `4e-3`, weight decay `0.05`, batch size `4096`, 300 epochs, 20 warmup epochs, cosine decay, RandAugment `(9, 0.5)`, Mixup `0.8`, Cutmix `1.0`, Random Erasing `0.25`, label smoothing `0.1`, LayerScale init `1e-6`, EMA `0.9999`, and stochastic-depth rates `0.1/0.4/0.5/0.5` for T/S/B/L. For ImageNet-22K pretraining, EMA is off and stochastic-depth rates are `0.0/0.0/0.1/0.1/0.2` for T/S/B/L/XL.
