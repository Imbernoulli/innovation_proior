# Self-attention blocks in convolutional image generators (full-attention placement)

## Problem

A convolutional generator/denoiser reproduces local image statistics (texture, color, edges) well
but botches long-range geometry, because convolution has a local receptive field: two distant
regions are coupled only through a long chain of local layers, which is costly to represent, hard to
optimize, and statistically brittle. Enlarging kernels buys reach only by sacrificing the efficiency
of local convolution and still imposes a fixed-shape (not content-addressed) neighborhood.

## Key idea

Insert a **self-attention block** into the convolutional backbone at the spatial feature maps. Every
output location is computed as a content-weighted average over **all** input locations, so any two
positions are coupled in a single layer. The block is the non-local / scaled-dot-product
self-attention operation adapted to a 2-D feature map: query/key/value are per-position linear maps
(1×1 convolutions), the attention weights are a softmax over learned dot-product affinities between
positions, and the output is wrapped in a residual that is **identity at initialization** (a
learnable scale `γ` initialized to 0, or equivalently a zero-initialized output projection) so the
block can be added to a working network without disturbing it and grows its influence only as it
helps.

The **full-attention** variant places this block at **every resolution** of the UNet
denoiser/generator (every down- and up-block, interleaved with the convolutional residual blocks),
rather than only at one chosen resolution — the maximally global placement, at the cost of paying the
`O(N²)` attention cost (`N = H·W`) at the high-resolution maps too.

## The self-attention module

For a feature map `x ∈ R^{C×N}` with `N = H*W` spatial locations, define per-position projections
`q_i = W_q x_i`, `k_j = W_k x_j`, and `v_j = W_v x_j`.

- Affinity logits in the scaled-dot-product form: `l_{ij} = q_i^T k_j / sqrt(d)`.
- Attention weights: `a_{ij} = exp(l_{ij}) / sum_{j'=1}^N exp(l_{ij'})`, a softmax over keys `j`.
- Attended output: `o_i = sum_j a_{ij} v_j`.
- Residual output: `y_i = x_i + gamma * W_o o_i`, with `gamma = 0` at initialization; in the DDPM
  denoiser form, the same identity-at-init invariant is implemented by a zero-initialized output
  projection. The embedded-Gaussian SAGAN code uses the same softmax-weighted value average with
  scale `1`; the scaled form is the DDPM/diffusers convention.

Design choices and their reasons:
- **1×1 convolutions or per-token linear layers** implement `W_q`, `W_k`, `W_v`, and `W_o`: channel
  mixing happens independently at each spatial location; spatial mixing is reserved for attention.
- **Softmax over keys** gives nonnegative weights summing to one for each query `i`, so `o_i` is a
  convex combination of value vectors before the output projection.
- **`1/sqrt(d)` scaling** follows from the variance calculation: a dot product of `d` independent
  mean-zero, variance-one component products has variance `d`; dividing by `sqrt(d)` keeps logits in
  the softmax's responsive range.
- **Bottlenecked generator block** uses `C/8` channels for query/key and `C/2` for values, with an
  output projection back to `C`; optional key/value max-pooling reduces the `N^2` pairwise map.
- **Diffusion denoiser block** uses GroupNorm before q/k/v projection and a residual `Attention`
  processor over flattened spatial tokens. In diffusers `AttnDownBlock2D` / `AttnUpBlock2D`, each
  residual block is followed by `Attention(..., norm_num_groups=32, residual_connection=True,
  upcast_softmax=True)`. `UNet2DModel` defaults to `attention_head_dim=8` unless overridden, so a
  `C`-channel map uses `C/8` heads of width 8. The stock diffusers `Attention` constructor adds the
  residual path but does not add a `gamma=0` gate or zero-initialize `to_out` by default.

## Working code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMapBlock(nn.Module):
    """Bottlenecked all-position attention over a [B, C, H, W] feature map."""

    def __init__(self, channels, pool_keys_values=True, scale_logits=True):
        super().__init__()
        qk_channels = max(1, channels // 8)
        value_channels = max(1, channels // 2)
        self.pool_keys_values = pool_keys_values
        self.scale = qk_channels ** -0.5 if scale_logits else 1.0

        self.to_q = nn.Conv2d(channels, qk_channels, 1)
        self.to_k = nn.Conv2d(channels, qk_channels, 1)
        self.to_v = nn.Conv2d(channels, value_channels, 1)
        self.proj_out = nn.Conv2d(value_channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.to_q(x).flatten(2).transpose(1, 2)   # [B, H*W, C/8]

        k_map = self.to_k(x)
        v_map = self.to_v(x)
        if self.pool_keys_values:
            k_map = F.max_pool2d(k_map, kernel_size=2, stride=2)
            v_map = F.max_pool2d(v_map, kernel_size=2, stride=2)

        k = k_map.flatten(2).transpose(1, 2)          # [B, M, C/8]
        v = v_map.flatten(2).transpose(1, 2)          # [B, M, C/2]

        logits = torch.einsum("bid,bjd->bij", q, k) * self.scale
        weights = logits.softmax(dim=-1)              # softmax over key/value positions j
        out = torch.einsum("bij,bjd->bid", weights, v)
        out = out.transpose(1, 2).reshape(B, v.shape[-1], H, W)
        return x + self.gamma * self.proj_out(out)    # gamma=0 gives identity at init
```

The MLS-Bench full-attention variant uses diffusers' own denoiser attention blocks and changes only
where they are placed:

```python
import os
from diffusers import UNet2DModel


def build_model(device):
    channels = (128, 256, 256, 256)
    if os.environ.get("BLOCK_OUT_CHANNELS"):
        channels = tuple(int(x) for x in os.environ["BLOCK_OUT_CHANNELS"].split(","))
    layers = int(os.environ.get("LAYERS_PER_BLOCK", 2))

    return UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        block_out_channels=channels,
        down_block_types=("AttnDownBlock2D",) * len(channels),
        up_block_types=("AttnUpBlock2D",) * len(channels),
        layers_per_block=layers,
        attention_head_dim=8,
        norm_num_groups=32,
        norm_eps=1e-6,
        act_fn="silu",
        time_embedding_type="positional",
        flip_sin_to_cos=False,
        freq_shift=1,
        downsample_padding=0,
    ).to(device)
```

Here `AttnDownBlock2D` / `AttnUpBlock2D` interleave convolutional residual blocks with diffusers
`Attention` modules. Selecting them at all four resolution levels is the full-attention placement.
