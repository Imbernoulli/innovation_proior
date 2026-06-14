**Problem (from rung 3).** The placement ladder converged: standard (FID 18.60/8.47/7.18) beats
full-attn at medium/large, full-attn (17.44/8.82/7.31) holds small. A single on/off attention tuple
can't satisfy both ends — broad attention helps the thin small model, efficient attention helps the
large one. The ladder only ever varied attention *placement*; it froze the residual-block internals,
the head count, and the timestep injection. Those frozen internals are the remaining headroom.

**Key idea (ADM, the improved diffusion UNet).** Improve the things placement held constant, so
attention can be afforded more broadly without full-attn's cost:
- **Multi-resolution attention at the 3 finer tiers (32/16/8), conv-only at 4×4.** Broader than
  standard's 16×16-only (the breadth full-attn's small-scale win showed helps a thin model),
  disciplined unlike full-attn's uniform everywhere.
- **Multi-head attention, fixed 64 channels per head** (`attention_head_dim=64`). A single head
  forces one relevance pattern; fixing channels-per-head keeps a constant comparison subspace and
  grows the head count on wider maps — strictly more expressive at the same FLOPs.
- **Adaptive group normalization** (`resnet_time_scale_shift="scale_shift"`, which the library maps
  to `AdaGroupNorm`): the timestep embedding produces a per-channel scale *and* shift of the
  normalized activations (`(1+scale)·GroupNorm(h)+shift`), a FiLM-style gain control, vs the baselines'
  additive-only injection. Identity-centered at init.
- **Dropout 0.1** in the residual blocks (standard regularizer for a same-resolution image backbone).

**Why it beats the placement-only baselines.** It gets the small-scale benefit of broad attention and
the large-scale benefit of efficient attention at once (multi-head 64-channel attention is cheaper
per unit coherence), and AdaGN gives the shared denoiser per-channel gain control across the noise
schedule that an additive bias cannot.

**Scaffold edit.** A `UNet2DModel` with attention-bearing down/up blocks at the three finer
resolutions, `attention_head_dim=64`, `resnet_time_scale_shift="scale_shift"`, `dropout=0.1`;
everything else the shared DDPM config. The fixed substrate (ε-MSE, linear 1000-step schedule, 50-step
DDIM, AdamW, EMA, clean-fid) is untouched.

**Harness caveat.** ADM's single biggest published architectural win — BigGAN-style residual blocks
that fold up/downsampling into the residual path — is *not* a `UNet2DModel` knob, so it is omitted
here; this finale lands the FID-relevant subset the scaffold can express (multi-resolution multi-head
attention + AdaGN). The 1/√2 residual rescale is deliberately not used (redundant with group norm).

**Bar to clear (no run yet).** Must beat the best baseline at each scale: under 17.44 small (full-attn),
under 8.47 medium and 7.18 large (standard). Expect the clearest win at small (breadth + cheap heads),
a thinner edge at medium/large where standard is near the placement optimum and the gain must come
from AdaGN and multi-head; the risk is the 32×32-tier attention reintroducing over-provisioning cost
at large scale.

```python
def build_model(device):
    """ADM-style improved UNet: multi-resolution multi-head attention + AdaGN conditioning."""
    channels = (128, 256, 256, 256)
    if os.environ.get('BLOCK_OUT_CHANNELS'):
        channels = tuple(int(x) for x in os.environ['BLOCK_OUT_CHANNELS'].split(','))
    layers = int(os.environ.get('LAYERS_PER_BLOCK', 2))

    return UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        block_out_channels=channels,
        # attention at the 3 finer feature tiers (32, 16, 8); pure conv at the 4x4 bottleneck tier
        down_block_types=("AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
        layers_per_block=layers,
        attention_head_dim=64,                     # multi-head: num_heads = channels // 64
        resnet_time_scale_shift="scale_shift",     # AdaGN (FiLM-style per-channel scale + shift)
        dropout=0.1,
        norm_num_groups=32,
        norm_eps=1e-6,
        act_fn="silu",
        time_embedding_type="positional",
        flip_sin_to_cos=False,
        freq_shift=1,
        downsample_padding=0,
    ).to(device)
```
