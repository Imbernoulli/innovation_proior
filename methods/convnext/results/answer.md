# ConvNeXt

## Problem

Hierarchical vision Transformers (e.g. Swin) had displaced ConvNets as the default vision backbone, and the performance gap was widely attributed to self-attention. But the comparison conflated many differences at once — training recipe, macro layout, and micro design — making it impossible to say whether attention was the cause. ConvNeXt isolates the question by holding FLOPs fixed and modernizing a standard ResNet toward a Swin Transformer one decision at a time, using no attention modules, and shows a pure ConvNet can match and exceed Swin.

## Key idea

Take the differences between a textbook ResNet-50 and Swin-T and transplant the non-attention ones into the ConvNet, measuring ImageNet-1K accuracy after each (ResNet-50 / Swin-T regime, ~4.5 GFLOPs):

1. **Modern training recipe** (held fixed afterward): AdamW, 300 epochs, linear warmup + cosine decay, Mixup/Cutmix/RandAugment/Random Erasing, label smoothing, stochastic depth, EMA. 76.1 → 78.8.
2. **Stage compute ratio** (3,4,6,3) → (3,3,9,3), matching Swin's 1:1:3:1. → 79.4.
3. **Patchify stem**: replace 7×7 stride-2 conv + maxpool with a 4×4 stride-4 non-overlapping conv. → 79.5.
4. **Depthwise conv + width**: make the spatial conv depthwise (spatial-only mixing, paired with 1×1 channel mixing — the attention/MLP split), widen 64 → 96. → 80.5.
5. **Inverted bottleneck**: MLP-style expand-4×-then-contract (96→384→96). → 80.6.
6. **Move depthwise conv up** (token-mixer first, on narrow channels), enabling a **large 7×7 kernel** (saturates at 7). → 80.6.
7. **Micro**: ReLU → GELU; keep one activation and one normalization per block; BatchNorm → LayerNorm. → 81.5.
8. **Separate downsampling**: 2×2 stride-2 conv between stages, with a LayerNorm before each downsample, after the stem, and after global pool. → 82.0, surpassing Swin-T's 81.3.

The resulting block: depthwise 7×7 conv → LayerNorm → 1×1 (×4 expand) → GELU → 1×1 (contract) → per-channel LayerScale → stochastic-depth → residual add. The macro net: patchify stem → 4 stages of these blocks with widths doubling per stage → separate downsamplers → global average pool → LayerNorm → linear head.

Variants differ only in depths and widths:
- ConvNeXt-T: dims (96,192,384,768), depths (3,3,9,3)
- ConvNeXt-S: dims (96,192,384,768), depths (3,3,27,3)
- ConvNeXt-B: dims (128,256,512,1024), depths (3,3,27,3)
- ConvNeXt-L: dims (192,384,768,1536), depths (3,3,27,3)
- ConvNeXt-XL: dims (256,512,1024,2048), depths (3,3,27,3)

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last (N,H,W,C) or channels_first (N,C,H,W)."""
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
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
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
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.):
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
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
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
        return self.head(x)

def convnext_tiny(**kwargs):
    return ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)

def convnext_small(**kwargs):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)

def convnext_base(**kwargs):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
```

Default ImageNet-1K training uses AdamW (lr 4e-3, weight decay 0.05, batch 4096), 300 epochs with 20-epoch linear warmup and cosine decay, the full augmentation/regularization stack, LayerScale init 1e-6, and per-variant stochastic depth rates (0.1/0.4/0.5/0.5 for T/S/B/L).
