# U-Net, distilled

The U-Net is a convolutional encoder-decoder for dense, same-size image-to-field
prediction. A *contracting path* of convolution-and-pool stages builds context and a large
receptive field (resolving *what* is at a pixel); a symmetric *expanding path* of
upsampling stages recovers full resolution (resolving *where*); and **skip connections at
every level concatenate the high-resolution encoder feature maps onto the matching decoder
feature maps**, so a subsequent convolution can learn to fuse fine localization with coarse
context. The two arms are equally deep and equally wide, giving the architecture its
U shape — the wide expanding path is what lets context propagate up to full resolution.

## Problem it solves

Predict an output for **every** pixel, at the input's spatial size, with precise
localization. The obstacle is structural: building a large receptive field (needed to
decide *what*) requires pooling/striding, which destroys the spatial resolution (needed to
decide *where*). U-Net resolves the tension by recovering the lost spatial detail from the
early high-resolution layers through concatenating skip connections, instead of trying to
recover it from the pooled-away deep features alone.

## Key idea

- **Contracting path:** repeat [two 3×3 convolutions + ReLU] then 2×2 downsampling;
  **double the channels** at each downsample (compensating the 4× spatial shrink to keep
  capacity roughly constant while abstraction grows).
- **Expanding path:** at each step, upsample, halve the channels, **concatenate the
  same-resolution encoder feature map**, then [two 3×3 convolutions + ReLU]. The expanding
  path is symmetric to the contracting one (same depth, comparable width) so it can carry
  context to high resolution.
- **Skip = concatenation, not addition,** at **every** resolution. Concatenation stacks the
  encoder's *where*-channels next to the decoder's *what*-channels and hands them to a
  learnable convolution, which decides how to mix them — rather than a fixed sum (as in
  FCN) that pre-commits the combination and discards most of the encoder's channels. The
  full rich feature maps are routed across, not thin class-score maps.
- **No fully-connected layers** anywhere, keeping the network fully convolutional and able
  to accept arbitrary input sizes.
- **Valid (unpadded) convolutions + center-crop before concat** in the original
  segmentation form: the output is smaller than the input by a fixed border, containing
  only pixels whose full context truly existed in the input. This enables the *overlap-tile*
  strategy — tile a huge image, mirror-pad the true borders, predict each valid interior,
  stitch seamlessly — so image size is not capped by GPU memory.

## Training in the few-data regime

- **Energy:** pixelwise softmax + weighted cross-entropy, written as the weighted
  log-likelihood `p_k(x) = exp(a_k(x)) / Σ_{k'} exp(a_{k'}(x))`,
  `E = Σ_x w(x)·log p_{ℓ(x)}(x)` (training drives `E` up; equivalently it minimizes `−E`).
- **Border-weighted loss** to separate touching same-class objects:
  `w(x) = w_c(x) + w_0·exp(−(d_1(x)+d_2(x))²/(2σ²))`, with `w_c` balancing class
  frequencies and the exponential term (large where `d_1+d_2` is small, i.e. in the thin
  gap between two nearby cells) forcing the network to learn separating borders;
  `w_0 = 10`, `σ ≈ 5` px.
- **Elastic-deformation data augmentation** (random smooth warps; displacements on a coarse
  grid, Gaussian σ = 10 px, interpolated) — the dominant tissue variation — lets the
  network learn deformation invariance from ~30 images; dropout at the deep end adds
  implicit augmentation.
- **Initialization** `N(0, √(2/N))`, N = fan-in (e.g. 3×3 conv over 64 channels → N = 576),
  keeps each feature map at ~unit variance through many conv+ReLU layers and multiple paths.
- **Optimization:** SGD, batch size 1 (favoring large input tiles over a large batch to use
  GPU memory), high momentum 0.99 (so many recent single-image gradients form each update).

## Field-appropriate final form: U-Net as a same-size image-to-field predictor

The same encoder-decoder-with-concatenating-skips is the standard backbone well beyond
segmentation. As a denoiser that maps a noisy image + scalar level to a same-size field, the
architecture is adapted in a few principled ways while leaving the U-topology and per-level
concatenating skips untouched:

- **Padded ("same") convolutions, no cropping.** The output must match the input size
  exactly (no shrinking border), and the overlap-tile rationale for valid convolutions does
  not apply to a fixed small input — so every 3×3 conv pads by 1 and skips concatenate
  already-aligned maps.
- **Per-block conditioning on a scalar level.** A sinusoidal embedding of the scalar
  (multi-frequency sin/cos, the sequence-model positional family) passes through a small MLP
  and is **added into every block**, so one shared network handles all levels.
- **Residual conv blocks** (two convs as a correction over a shortcut) to keep a deep
  network trainable; **group normalization** (statistics over channel groups within one
  example, batch-independent → stable at batch size 1) with a SiLU nonlinearity.
- **Learned strided-conv downsampling** (instead of fixed max-pool) and **nearest-neighbor
  upsampling + conv** (avoiding transposed-conv checkerboard artifacts).
- **No per-resolution self-attention:** `DownBlock2D` and `UpBlock2D` contain only
  residual convolutional blocks plus sampling layers — no attention at any resolution. A
  single self-attention layer survives only at the bottleneck (`UNetMidBlock2D` with its
  default `add_attention=True`), where the grid is smallest (e.g. 4×4) and all-to-all
  attention — whose cost grows with the square of the number of spatial positions — is
  cheap. This tests whether convolution's pooled receptive field and skip-routed detail
  suffice for global coherence at the per-resolution stages.

## Working code

A faithful U-Net denoiser with no per-resolution self-attention, structured like the
established implementation with `DownBlock2D`, `UpBlock2D`, `ResnetBlock2D`, and the default
`UNetMidBlock2D` (one self-attention at the bottleneck): residual blocks with timestep
injection, GroupNorm/SiLU, strided-conv down, nearest-up, concatenating per-level skips, and
self-attention only at the bottleneck. It returns a same-size output field.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps, dim, flip_sin_to_cos=False,
                           downscale_freq_shift=1.0, max_period=10000):
    """Sinusoidal embedding of a scalar level (DDPM/Transformer positional sinusoid)."""
    assert timesteps.ndim == 1
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32,
                                                     device=timesteps.device)
    exponent = exponent / (half - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class ResnetBlock2D(nn.Module):
    """GroupNorm -> SiLU -> conv1 -> (+ time proj) -> GroupNorm -> SiLU -> dropout
    -> conv2 -> + (1x1) shortcut. Padded 3x3 convs preserve spatial size."""
    def __init__(self, in_ch, out_ch, temb_ch, groups=32, eps=1e-6, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch, eps=eps)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = nn.Linear(temb_ch, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch, eps=eps)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv_shortcut = (nn.Conv2d(in_ch, out_ch, kernel_size=1)
                              if in_ch != out_ch else nn.Identity())

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.conv_shortcut(x)


class Downsample2D(nn.Module):
    """Learned downsampling: stride-2 3x3 conv (with asymmetric pad for clean halving)."""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class Upsample2D(nn.Module):
    """Nearest-neighbor upsample then 3x3 conv (no checkerboard)."""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class DownBlock2D(nn.Module):
    """layers_per_block residual blocks, then optional learned downsample.
    Pure convolutional: no self-attention."""
    def __init__(self, in_ch, out_ch, temb_ch, num_layers=2,
                 add_downsample=True, groups=32, eps=1e-6, dropout=0.0):
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, temb_ch, groups, eps, dropout)
             for i in range(num_layers)])
        self.downsamplers = nn.ModuleList([Downsample2D(out_ch)]) if add_downsample else None

    def forward(self, x, temb):
        out = ()
        for resnet in self.resnets:
            x = resnet(x, temb)
            out += (x,)
        if self.downsamplers is not None:
            for d in self.downsamplers:
                x = d(x)
            out += (x,)
        return x, out


class UpBlock2D(nn.Module):
    """Concatenate the matching encoder skip onto the running map, fuse with a
    residual block (learnable fusion of 'where' + 'what'); then optional upsample.
    num_layers = layers_per_block + 1 (one block per popped skip)."""
    def __init__(self, in_ch, prev_out_ch, out_ch, temb_ch, num_layers=3,
                 add_upsample=True, groups=32, eps=1e-6, dropout=0.0):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_skip_ch = in_ch if i == num_layers - 1 else out_ch
            resnet_in_ch = prev_out_ch if i == 0 else out_ch
            resnets.append(ResnetBlock2D(resnet_in_ch + res_skip_ch, out_ch,
                                         temb_ch, groups, eps, dropout))
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = nn.ModuleList([Upsample2D(out_ch)]) if add_upsample else None

    def forward(self, x, res_samples, temb):
        for resnet in self.resnets:
            skip = res_samples[-1]
            res_samples = res_samples[:-1]
            x = torch.cat([x, skip], dim=1)        # concat, not add
            x = resnet(x, temb)
        if self.upsamplers is not None:
            for u in self.upsamplers:
                x = u(x)
        return x


class UNetMidBlock2D(nn.Module):
    """Bottleneck: resnet -> one self-attention -> resnet. The lone global mixer
    the pure-convolutional variant keeps (add_attention=True by default), at the
    smallest resolution where all-to-all attention is cheap."""
    def __init__(self, ch, temb_ch, groups=32, eps=1e-6, dropout=0.0, num_heads=1):
        super().__init__()
        self.resnet1 = ResnetBlock2D(ch, ch, temb_ch, groups, eps, dropout)
        self.norm = nn.GroupNorm(groups, ch, eps=eps)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)
        self.resnet2 = ResnetBlock2D(ch, ch, temb_ch, groups, eps, dropout)

    def forward(self, x, temb):
        x = self.resnet1(x, temb)
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).transpose(1, 2)   # [B, HW, C]
        h, _ = self.attn(h, h, h)
        x = x + h.transpose(1, 2).reshape(B, C, H, W)           # residual
        return self.resnet2(x, temb)


class ConvUNet(nn.Module):
    """Pure-convolutional U-Net denoiser with NO per-resolution self-attention
    (the lone self-attention survives only at the bottleneck): contracting
    residual-conv path with channel-doubling + learned downsampling, a
    bottleneck, and a symmetric expanding path with per-level concatenating skips."""
    def __init__(self, in_channels=3, out_channels=3,
                 block_out_channels=(128, 256, 256, 256), layers_per_block=2,
                 norm_num_groups=32, norm_eps=1e-6, dropout=0.0,
                 flip_sin_to_cos=False, freq_shift=1):
        super().__init__()
        C0 = block_out_channels[0]
        temb_ch = C0 * 4
        self.proj_dim, self.flip_sin_to_cos, self.freq_shift = C0, flip_sin_to_cos, freq_shift
        self.time_mlp = nn.Sequential(nn.Linear(C0, temb_ch), nn.SiLU(),
                                      nn.Linear(temb_ch, temb_ch))
        self.conv_in = nn.Conv2d(in_channels, C0, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        out_ch = C0
        for i in range(len(block_out_channels)):
            in_ch = out_ch
            out_ch = block_out_channels[i]
            self.down_blocks.append(
                DownBlock2D(in_ch, out_ch, temb_ch, layers_per_block,
                            add_downsample=(i != len(block_out_channels) - 1),
                            groups=norm_num_groups, eps=norm_eps, dropout=dropout))

        self.mid_block = UNetMidBlock2D(block_out_channels[-1], temb_ch,
                                        norm_num_groups, norm_eps, dropout)

        self.up_blocks = nn.ModuleList()
        rev = list(reversed(block_out_channels))
        out_ch = rev[0]
        for i in range(len(block_out_channels)):
            prev_out_ch = out_ch
            out_ch = rev[i]
            in_ch = rev[min(i + 1, len(block_out_channels) - 1)]
            self.up_blocks.append(
                UpBlock2D(in_ch, prev_out_ch, out_ch, temb_ch, layers_per_block + 1,
                          add_upsample=(i != len(block_out_channels) - 1),
                          groups=norm_num_groups, eps=norm_eps, dropout=dropout))

        self.conv_norm_out = nn.GroupNorm(norm_num_groups, C0, eps=norm_eps)
        self.conv_out = nn.Conv2d(C0, out_channels, kernel_size=3, padding=1)

    def forward(self, sample, timestep):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        timestep = timestep * torch.ones(sample.shape[0], dtype=timestep.dtype,
                                         device=sample.device)
        t_emb = get_timestep_embedding(timestep, self.proj_dim,
                                       self.flip_sin_to_cos, self.freq_shift)
        emb = self.time_mlp(t_emb)

        sample = self.conv_in(sample)
        res = (sample,)
        for blk in self.down_blocks:
            sample, r = blk(sample, emb)
            res += r

        sample = self.mid_block(sample, emb)

        for blk in self.up_blocks:
            n = len(blk.resnets)
            r, res = res[-n:], res[:-n]
            sample = blk(sample, r, emb)

        sample = self.conv_out(F.silu(self.conv_norm_out(sample)))
        return sample
```

## Relation to prior methods

- **Sliding-window patch classifier (Ciresan et al., 2012):** replaced entirely — U-Net
  produces the full map in one forward pass and decouples context from localization (no
  patch-size knob trading the two).
- **FCN (Long, Shelhamer & Darrell, 2014):** U-Net keeps the fully-convolutional
  inference and learnable in-network upsampling, but routes the **full** high-resolution
  encoder feature maps across at **every** level and **concatenates** them for a learnable
  fusion (vs. FCN's addition of thin class-score maps at two or three points), and makes
  the expanding path deep and wide so context reaches full resolution.
- **Multi-layer feature classifiers (Hypercolumns; cascaded hierarchical models):** the
  what+where combination becomes a single end-to-end network with a deep learned expansive
  decoder, rather than a per-pixel classifier over hand-stacked multi-layer features.
