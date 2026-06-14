## Research question

The task is to design the denoising network for an unconditional CIFAR-10 diffusion model so that
it produces lower-FID samples than the standard DDPM-style UNet, under a training pipeline that is
completely fixed except for the architecture. The diffusion target (ε-prediction with MSE), the
noise schedule (linear, 1000 steps), the sampler (50-step DDIM), the optimizer (AdamW, lr 2e-4),
the EMA (0.9995), and the FID evaluation (clean-fid against the 50k CIFAR-10 train set, lower is
better) are all frozen. The single thing being designed is the `build_model(device)` function: the
`nn.Module` that maps a noised image and a timestep to a same-shaped noise prediction. So this is a
pure architecture-design question — what residual blocks, what normalization, and above all *where
to place self-attention* across the UNet's resolution levels — judged on whether the resulting
denoiser generates sharper, more coherent 32×32 images.

## Prior art before the first rung

The denoiser is a UNet, and the design space is the lineage the first rung reacts to.

- **The UNet (Ronneberger et al., 2015).** An encoder–decoder with skip connections: a contracting
  path of convolution-and-downsample stages builds a large receptive field, a symmetric expanding
  path of upsample stages recovers resolution, and skip connections concatenate equal-resolution
  encoder maps onto the decoder so a convolution can fuse fine localization with coarse context.
  It is the natural backbone for a denoiser, whose input and output are both same-resolution images.
  Gap: pure convolution has a *local* receptive field, so coordinating structure between two distant
  regions of an image must be carried indirectly through a long chain of conv layers — costly to
  represent and brittle.
- **The DDPM denoiser (Ho et al., 2020, arXiv:2006.11239).** Instantiates the reverse process as a
  time-conditioned UNet over Wide-ResNet blocks: a sinusoidal timestep embedding added into every
  residual block, group normalization, and a single global self-attention layer at the 16×16 feature
  resolution. This is the architecture the `google/ddpm-cifar10-32` configuration uses, and the
  reference point for "standard." Gap: attention placement is a single fixed choice (only 16×16) that
  was never swept on this exact harness — whether more, less, or differently-placed attention helps
  CIFAR-10 FID under a fixed budget is open.
- **Self-attention as a feature-map operation (Wang et al. 2018 non-local; Vaswani et al. 2017;
  Zhang et al. 2019 SAGAN).** Self-attention couples any two spatial positions in one layer with a
  content-dependent weight — the operation convolution cannot do locally. Its cost is O(N²) in the
  number of spatial positions N = H·W, so where it is placed across the UNet's resolution levels is
  itself the design lever. Gap: the cost is prohibitive at the largest feature maps and the global
  structure to coordinate is thin at the coarsest, so "attention everywhere" and "attention nowhere"
  are both plausible and untested extremes here.

## The fixed substrate

A self-contained unconditional DDPM training script (`custom_train.py`) is frozen and must not be
touched. It maps CIFAR-10 images to [−1, 1] with random horizontal flips; at each step it draws a
random timestep t, forms the noised image `x_t = add_noise(x_0, ε, t)` under a linear 1000-step
schedule, predicts ε with the model, and minimizes `F.mse_loss(pred, ε)` (ε-prediction is fixed and
not editable). It runs AdamW (lr 2e-4, weight decay 1e-4) with gradient clipping and a parameter EMA
(rate 0.9995), trains across three channel scales — Small `(64,128,128,128)` ~9M params batch 128,
Medium `(128,256,256,256)` ~36M params batch 128, Large `(256,512,512,512)` ~140M params batch 64 —
and at evaluation samples with 50-step DDIM and scores FID with clean-fid against the 50k CIFAR-10
train set. The channel widths arrive through the `BLOCK_OUT_CHANNELS` environment variable and
`LAYERS_PER_BLOCK` (default 2), so one architecture definition scales across all three tiers.

## The editable interface

Exactly one region is editable — the `build_model(device)` function in `custom_train.py`
(lines 31–58 of the scaffold). It must return an `nn.Module` satisfying the denoiser contract:

- **Input:** `(x, timestep)` with `x` of shape `[B, 3, 32, 32]` and `timestep` of shape `[B]`.
- **Output:** an object with a `.sample` attribute of shape `[B, 3, 32, 32]` — the predicted ε.

`UNet2DModel` from `diffusers` already satisfies this interface; a fully custom `nn.Module` is also
allowed. Every method on the ladder is a fill of this one function — the same UNet wiring, residual
blocks, group norm, sinusoidal timestep embedding, and channel schedule, differing only in which
resolution levels carry self-attention (`AttnDownBlock2D`/`AttnUpBlock2D` vs `DownBlock2D`/
`UpBlock2D`) and the attention configuration. The starting point is the scaffold default, an
unimplemented stub; each method replaces exactly this function and nothing else.

```python
# EDITABLE region of custom_train.py — default fill (stub)
def build_model(device):
    """Build a UNet model for unconditional CIFAR-10 diffusion.

    The model must satisfy:
    - Input:  (x, timestep) where x is [B, 3, 32, 32], timestep is [B]
    - Output: object with .sample attribute of shape [B, 3, 32, 32]
    - UNet2DModel from diffusers satisfies this interface

    Channel widths arrive via env var BLOCK_OUT_CHANNELS (e.g. "128,256,256,256");
    LAYERS_PER_BLOCK (default 2) is also available. Available block types:
        "DownBlock2D"     / "UpBlock2D"      (pure convolution)
        "AttnDownBlock2D" / "AttnUpBlock2D"  (conv + self-attention)
    Other knobs: layers_per_block, norm_num_groups, attention_head_dim,
                 resnet_time_scale_shift, act_fn, etc.
    """
    raise NotImplementedError("Implement build_model")
```

## Evaluation settings

Each candidate architecture is trained from scratch at the three channel scales above (Small,
Medium, Large), each at a fixed step budget per scale, on CIFAR-10 (32×32, unconditional), with a
single seed (42). Samples are drawn with 50-step DDIM and scored by FID (clean-fid, 50k samples,
against the CIFAR-10 train set); lower FID is better. The reported metrics are the best FID at each
scale — `best_fid_small`, `best_fid_medium`, `best_fid_large` — so an architecture is judged on
whether it transfers across the budget range, not on a single tier. Improvements must come from the
architecture alone; the data, loss target, noise schedule, optimizer, sampler, and evaluation are
all held fixed.
