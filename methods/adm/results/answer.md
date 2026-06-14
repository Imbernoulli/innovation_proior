# ADM: improved diffusion UNet

ADM is the architecture-side redesign of the diffusion denoising UNet. It keeps the
epsilon-prediction objective, learned-variance hybrid loss, DDPM/DDIM sampling contract,
optimizer, and EMA machinery fixed, and changes only the network that maps `(x_t, t)` to
the predicted noise. A separate classifier-guidance path changes sampling and is outside
this trace.

## Problem it solves

Diffusion models are stable likelihood-based image generators, but on large diverse image
datasets their sample-quality metrics lag the best GAN baselines. The denoiser being used
is still close to the CIFAR-era UNet: single-head attention at one feature resolution,
additive timestep injection, and resampling outside the residual blocks. The architecture
question is whether a better denoiser can recover quality while preserving the diffusion
training and sampling setup.

## Architecture decisions

- **Adopt multi-resolution attention.** Use attention at feature resolutions 32, 16, and 8
  instead of only at 16, so long-range coordination is available at several coarse tiers
  where the quadratic attention cost is still tractable.
- **Adopt more attention heads.** Use a fixed `num_head_channels=64`, so
  `num_heads = channels // num_head_channels`; this keeps each head's comparison
  dimension fixed as the channel width changes.
- **Adopt BigGAN up/downsampling residual blocks.** Put resolution changes inside
  residual blocks, with the residual branch and skip path resampled in parallel, instead
  of using a standalone resample between residual stages.
- **Adopt AdaGN conditioning.** The target formula is
  `AdaGN(h, y) = y_s * GroupNorm(h) + y_b`. In the canonical implementation the embedding
  projection is split into `scale, shift` and applied in the residual block's `out_layers`
  as `GroupNorm(h) * (1 + scale) + shift`, so the modulation is identity-centered at
  initialization.
- **Do not adopt extra depth.** Increasing depth at roughly fixed model size helps FID but
  loses on wall-clock time to a wider model.
- **Do not adopt `1/sqrt(2)` residual rescaling.** It hurts FID in this normalized denoiser,
  so the final architecture drops it.

## Final default architecture

Use width as the capacity knob, two residual blocks per resolution, attention at
32/16/8 feature resolutions, `num_head_channels=64`, BigGAN-style residual up/downsampling,
and AdaGN scale-shift conditioning from the timestep embedding and optional class
embedding. Use the same diffusion loss and sampler around it.

## Canonical guided-diffusion block structure

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def normalization(channels, groups=32):
    return nn.GroupNorm(min(groups, channels), channels)


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = conv3(channels, self.out_channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.use_conv else x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.op = conv3(channels, self.out_channels, stride=2) if use_conv else nn.AvgPool2d(2, 2)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels, dropout, out_channels=None,
                 use_conv=False, use_scale_shift_norm=True, up=False, down=False):
        super().__init__()
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.updown = up or down

        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(), conv3(channels, self.out_channels)
        )
        if up:
            self.h_upd = Upsample(channels, use_conv=False)
            self.x_upd = Upsample(channels, use_conv=False)
        elif down:
            self.h_upd = Downsample(channels, use_conv=False)
            self.x_upd = Downsample(channels, use_conv=False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(conv3(self.out_channels, self.out_channels)),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv3(channels, self.out_channels)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = torch.einsum("bts,bcs->bct", weight, v)
        return out.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_head_channels=64):
        super().__init__()
        assert channels % num_head_channels == 0
        self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x.reshape(b, c, h * w)
        out = self.attention(self.qkv(self.norm(x_in)))
        out = self.proj_out(out)
        return (x_in + out).reshape(b, c, h, w)
```

## As a diffusers `UNet2DModel`

`UNet2DModel` can express the multi-resolution attention placement, fixed
channels-per-head, dropout, and AdaGN/AdaGroupNorm-style scale-shift conditioning. The
scale-shift path is selected by `resnet_time_scale_shift="scale_shift"` and implements
`GroupNorm(h) * (1 + time_scale) + time_shift` in `ResnetBlock2D`; attention blocks use
`heads = channels // attention_head_dim`, so `attention_head_dim=64` matches the 64
channels-per-head setting. The standard diffusers up/down blocks are not the exact
guided-diffusion BigGAN-in-residual-block variant.

```python
from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=128,
    in_channels=3,
    out_channels=3,
    block_out_channels=(256, 256, 512, 768, 1024),
    down_block_types=(
        "DownBlock2D", "DownBlock2D",
        "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D",
        "UpBlock2D", "UpBlock2D",
    ),
    layers_per_block=2,
    attention_head_dim=64,
    resnet_time_scale_shift="scale_shift",
    norm_num_groups=32,
    norm_eps=1e-6,
    dropout=0.0,
)
```
