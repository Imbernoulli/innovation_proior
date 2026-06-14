**Problem.** The whole task is architecture design for a CIFAR-10 diffusion denoiser, with the loss,
schedule, sampler, optimizer, and FID evaluation all fixed. Before measuring whether self-attention
earns its O(N²) cost, I need a floor: the leanest convolution-only denoiser the contract accepts.

**Key idea (the floor).** A time-conditioned UNet is the natural denoiser backbone — same-resolution
image in, same-resolution ε out, with skip connections so the fine detail a denoiser must restore
bypasses the bottleneck. The lean hypothesis is that *per-resolution* self-attention is unnecessary
on CIFAR-10: strip attention from every down/up stage (`DownBlock2D`/`UpBlock2D` at all four
resolutions). `UNet2DModel` still builds its default `UNetMidBlock2D`, which keeps one self-attention
layer at the 4×4 bottleneck — so a single global mixing layer survives where N² is trivial, and
nowhere it would hurt. This is the smallest, fastest architecture on the ladder and the floor every
later rung must beat to justify adding attention back.

**Why it works (or doesn't).** Convolution nails the local statistics CIFAR FID rewards (sharp edges,
texture, color); the UNet's pooling gives every output pixel a large receptive field and the
bottleneck attention coordinates the cheap global scale. What it is structurally worst at is
long-range coherence between distant regions, which must be carried indirectly through the conv stack.

**Scaffold edit / hyperparameters.** Fill `build_model` to return a `UNet2DModel` with pure-conv
down/up blocks; everything else is the shared DDPM config (`norm_num_groups=32`, `norm_eps=1e-6`,
SiLU, positional time embedding with `flip_sin_to_cos=False`, `freq_shift=1`, `downsample_padding=0`),
channels from `BLOCK_OUT_CHANNELS`, `layers_per_block` from the env (default 2). Only the
`down_block_types`/`up_block_types` tuple distinguishes this rung from the next.

**What to watch.** Expect competent, locally-coherent samples but the worst FID on the ladder, with
the largest gap to attention-bearing models at the larger channel scales where capacity is available.

```python
def build_model(device):
    """No-attention: pure convolutional UNet (no per-resolution attention)."""
    channels = (128, 256, 256, 256)
    if os.environ.get('BLOCK_OUT_CHANNELS'):
        channels = tuple(int(x) for x in os.environ['BLOCK_OUT_CHANNELS'].split(','))
    layers = int(os.environ.get('LAYERS_PER_BLOCK', 2))

    return UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        block_out_channels=channels,
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        layers_per_block=layers,
        norm_num_groups=32,
        norm_eps=1e-6,
        act_fn="silu",
        time_embedding_type="positional",
        flip_sin_to_cos=False,
        freq_shift=1,
        downsample_padding=0,
    ).to(device)
```
