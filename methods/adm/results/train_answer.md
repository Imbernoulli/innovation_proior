Diffusion models are attractive image generators because they are likelihood-based, trained with a single stationary mean-squared-error objective, and avoid the adversarial balancing act of GANs. By mid-2021 they already achieve state-of-the-art FID on CIFAR-10, but on larger and more diverse datasets such as ImageNet at 128×128 and 256×256 and LSUN categories they still trail the best GANs like BigGAN-deep. The training objective, the Gaussian noising process, and the DDPM or DDIM sampler all work well and should be kept fixed; the remaining suspect is the denoiser architecture itself. The prevailing UNet is essentially the one that made diffusion work on CIFAR-10: it uses a single self-attention head at a single feature resolution, injects the timestep embedding only additively, and performs downsampling and upsampling outside the residual blocks. Those choices were never stress-tested on large diverse images, where long-range structure exists at multiple scales and the conditioning signal must modulate a much deeper network. The question is whether redesigning only the denoiser can close the FID gap without changing anything else.

The answer is ADM, the Ablated Diffusion Model. It keeps the epsilon-prediction objective, the learned-variance hybrid loss, the optimizer, the EMA, and the sampler exactly as before, and changes only the network that maps a noisy image and timestep to a predicted noise tensor. The redesign is driven by targeted ablations on ImageNet 128×128, reading FID at two training checkpoints so that improvements are not transients. The core insight is to treat the denoiser as a design space and to keep only the changes that address specific weaknesses of the baseline UNet.

First, ADM places multi-head self-attention at multiple coarse feature resolutions. The baseline has attention only at 16×16, which forces all global coordination at other scales to propagate through many convolutional layers. ADM adds attention at the 32×32, 16×16, and 8×8 feature maps, leaving the finest map purely convolutional because attention is quadratic in the number of spatial positions and would be prohibitively expensive there. This gives the network direct long-range mixing at every scale where global structure is semantically important and the cost is still tractable.

Second, ADM uses multi-head attention with a fixed number of channels per head rather than a fixed head count. With num_head_channels set to 64, the number of heads grows as the channel width grows. This keeps the dimension of the query-key comparison fixed across resolutions, which is the standard Transformer convention and gives a cleaner scaling story than allowing per-head width to balloon on wide feature maps. Different heads can specialize to different spatial relations at the same time, and the total cost is comparable to a single full-width attention layer.

Third, resolution changes are folded into the residual blocks rather than placed between them. In the baseline, downsampling and upsampling are standalone pooling or interpolation operations with no residual identity path. ADM adopts BigGAN-style residual blocks in which the channel-changing convolution also resamples, while a parallel skip path resamples the identity connection. This makes every resolution transition a learned residual correction with a clean gradient path instead of a plain transform between residual stages.

Fourth, ADM replaces additive timestep injection with adaptive group normalization, or AdaGN. The timestep embedding is projected to per-channel scale and shift parameters that are applied after group normalization inside each residual block, using the form GroupNorm(h) * (1 + scale) + shift. The one-plus scale makes the modulation identity at initialization, so the conditioning grows in only as it proves useful. This gives the network multiplicative control over activations as a function of noise level, which is more expressive than a simple additive bias.

Two changes from the broader architectural toolkit are deliberately not adopted. Increasing depth at roughly fixed parameter count can improve final FID but loses on wall-clock time to target quality, so ADM spends its capacity budget on width instead. And the one-over-root-two residual-branch rescaling that helps some style-based generators and continuous-time score models actually hurts FID in this normalized denoiser, so it is omitted. The final architecture is therefore a wide UNet with two residual blocks per resolution, multi-resolution multi-head attention, BigGAN-style up/downsampling, and AdaGN conditioning.

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


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepBlock(nn.Module):
    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, TimestepBlock) else layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = conv3(channels, self.out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.use_conv else x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.op = conv3(channels, self.out_channels, stride=2) if use_conv else nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.op(x)


class ResBlock(TimestepBlock):
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


class Denoiser(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=128,
                 channel_mult=(1, 1, 2, 3, 4), num_res_blocks=2,
                 attention_resolutions=(32, 16, 8), num_head_channels=64,
                 dropout=0.0, image_size=128):
        super().__init__()
        emb_ch = base_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch)
        )
        self.base_channels = base_channels
        attention_ds = {image_size // r for r in attention_resolutions}

        ch = int(channel_mult[0] * base_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv3(in_channels, ch))])
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks = [ResBlock(ch, emb_ch, dropout, out_ch, use_scale_shift_norm=True)]
                ch = out_ch
                if ds in attention_ds:
                    blocks.append(AttentionBlock(ch, num_head_channels))
                self.input_blocks.append(TimestepEmbedSequential(*blocks))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(
                    ResBlock(ch, emb_ch, dropout, ch, use_scale_shift_norm=True, down=True)
                ))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, emb_ch, dropout, ch, use_scale_shift_norm=True),
            AttentionBlock(ch, num_head_channels),
            ResBlock(ch, emb_ch, dropout, ch, use_scale_shift_norm=True),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = base_channels * mult
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                blocks = [ResBlock(ch + ich, emb_ch, dropout, out_ch, use_scale_shift_norm=True)]
                ch = out_ch
                if ds in attention_ds:
                    blocks.append(AttentionBlock(ch, num_head_channels))
                if level != 0 and i == num_res_blocks:
                    blocks.append(ResBlock(ch, emb_ch, dropout, ch, use_scale_shift_norm=True, up=True))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*blocks))

        self.out = nn.Sequential(
            normalization(ch), nn.SiLU(), zero_module(conv3(ch, out_channels))
        )

    def forward(self, x, timestep):
        emb = self.time_embed(timestep_embedding(timestep, self.base_channels))
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)
```
