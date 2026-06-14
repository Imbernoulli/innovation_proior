## Research question

Class-conditional image generation with a diffusion denoiser: a UNet learns to predict the noise
$\epsilon_\theta(x_t, t, c)$ added to a CIFAR-10 image at noise level $t$, conditioned on the class
label $c$. The backbone, the noise schedule, the optimizer, the EMA, and the 50-step DDIM sampler are
all fixed. The **only** thing being designed is **how the class label is injected into the denoiser** —
the conditioning operator. Everything that turns "draw an image" into "draw an image *of this class*"
lives in one editable region, and the metric is FID against the CIFAR-10 train set (lower is better).

## Prior art before the first rung (conditioning-injection lineage)

The conditioning operators on this ladder are the resolution of a line of "how does side information
reach a convolutional feature stream" mechanisms. These precede the ladder; the fixed substrate below is
what they converged to.

- **Additive / concatenated conditioning (Conditional DCGAN; Conditional PixelCNN, van den Oord et al.
  2016; WaveNet).** Broadcast the conditioning vector into constant feature maps and concatenate before a
  conv, or add a conditioning-dependent bias. After the next linear map, `W[F; z] = W_F F + W_z z`, so the
  `z` term is a conditioning-dependent *additive bias* on the output. Gap: it can shift features but cannot
  rescale, amplify, gate, or negate them — only half the affine.
- **Feature-wise gates (LSTM gates; Squeeze-and-Excitation, Hu et al. 2018).** Multiply features by a
  bounded factor `sigmoid(g) ∈ (0,1)`. Gap: scale-only, bounded — attenuates or passes a channel but
  cannot amplify, shift a threshold, or negate; the additive half is missing.
- **Conditional / adaptive normalization (Conditional BatchNorm, de Vries et al. 2017; Conditional
  Instance Norm, Dumoulin et al. 2017; AdaIN, Huang & Belongie 2017).** A normalization layer ends in a
  per-channel affine `gamma · x_hat + beta`; make `(gamma, beta)` a function of the conditioning rather
  than learned constants. Gap as written: tied to a particular normalization (BN/IN) or to a discrete
  style table, and demonstrated on convolutional recognition / style transfer rather than on a denoiser
  with a timestep signal already occupying the affine.
- **FiLM (Perez et al., AAAI 2018).** The general statement: a conditioning input regresses a per-channel
  affine `gamma(z) · F + beta(z)`, with `gamma = 1 + Δγ` for identity init, applied anywhere, independent
  of resolution. It contains additive bias (`gamma = 1`) and gating (`beta = 0`, bounded `gamma`) as
  corners. Gap for this task: the affine is *spatially uniform and content-blind* — the same scale/shift
  at every position, drawn from the label alone.
- **Adaptive Group Norm (ADM, Dhariwal & Nichol 2021).** FiLM ported into a diffusion residual block: the
  combined timestep-and-class embedding linearly produces the GroupNorm scale/shift. This is the
  established socket by which a diffusion UNet already ingests the timestep. Gap: it took an *auxiliary
  classifier at sampling time* (classifier guidance) to get strong class adherence — the in-network affine
  alone was low-bandwidth for "which class."

## The fixed substrate

A self-contained class-conditional DDPM training script (`custom_train.py`) is frozen and must not be
touched. The denoiser is `ConditionalUNet`, which wraps a diffusers `UNet2DModel` (the
`google/ddpm-cifar10-32` architecture) and adds two conditioning hooks. The backbone runs at three
channel scales — Small `(64,128,128,128)` ~9M, Medium `(128,256,256,256)` ~36M, Large
`(256,512,512,512)` ~140M — chosen by env var, with `time_embed_dim = block_out_channels[0] * 4` and a
`class_embed = nn.Embedding(num_classes, time_embed_dim)`. Training: 35,000 steps per scale, AdamW
`lr=2e-4`, weight decay `1e-4`, grad-norm clip `1.0`, AMP, EMA rate `0.9995`, batch 128 (64 at Large).
Noise: diffusers `DDPMScheduler`, linear betas `1e-4 → 2e-2`, 1000 train steps, `epsilon` prediction,
`clip_sample=True`. The objective is the plain noise-MSE `F.mse_loss(pred_noise, noise)`. Evaluation: a
50-step DDIM sampler (`DDIMScheduler`, same betas) draws class-conditional samples scored by clean-fid
against the CIFAR-10 train statistics. A parameter budget of `1.05×` the cross-attention reference model
is enforced.

The substrate also provides two reusable conditioning utilities a method may call:

- `zero_module(m)` — zeros a module's parameters (for identity-at-init residual blocks).
- `CrossAttentionLayer(channels, context_dim, num_heads)` — GroupNorm, q from features / k,v from the
  context token(s), scaled-dot-product multi-head attention, **zero-initialized output projection**, and a
  residual add.
- `AdaLNBlock(channels, cond_dim)` — GroupNorm(1) then a **zero-initialized** `SiLU → Linear(cond_dim,
  3·channels)` producing scale/shift/gate, applied as `x + gate·((1+scale)·norm(x)+shift − x)` — adaptive
  LayerNorm-Zero on a `[B,C,H,W]` feature map.

The wiring is fixed: per forward, `ConditionalUNet` embeds the timestep (`unet.time_proj →
unet.time_embedding`) and the class (`class_embed`), calls `prepare_conditioning(time_emb, class_emb)` to
form the embedding `emb` fed to **every** residual block, then runs the UNet and after **each** down/mid/up
block applies the corresponding `ClassConditioner(h, class_emb)`.

## The editable interface

Exactly one region is editable — `prepare_conditioning(time_emb, class_emb)` and the `ClassConditioner`
class in `custom_train.py` (lines 195–227). Every method on the ladder is a fill of this same contract:

- `prepare_conditioning(time_emb, class_emb) -> [B, time_embed_dim]` — how the class embedding combines
  with the timestep embedding *before* the residual blocks (the AdaGN/time path).
- `ClassConditioner(channels, cond_dim)` with `forward(h, class_emb) -> [B,C,H,W]` — the module applied
  *after* each block (the post-block path: cross-attention, adaptive norm, or a no-op).

A method may route the class through the time path, the post-block path, or both. The denoising interface
`(x, timestep, class_id) → ε` and its shapes are invariant. The scaffold default is the unfilled contract:

```python
def prepare_conditioning(time_emb, class_emb):
    """Prepare the combined embedding used in ResBlocks.

    Args:
        time_emb:  [B, time_embed_dim] timestep embedding
        class_emb: [B, time_embed_dim] class embedding

    Returns: [B, time_embed_dim] embedding used in ResBlocks
    """
    raise NotImplementedError("Implement prepare_conditioning")


class ClassConditioner(nn.Module):
    """Conditioning module applied after each UNet block.

    Args (forward):
        h:         [B, C, H, W] feature map
        class_emb: [B, time_embed_dim] class embedding

    Available utilities:
        CrossAttentionLayer(channels, context_dim, num_heads)
        AdaLNBlock(channels, cond_dim)
    """
    def __init__(self, channels, cond_dim):
        super().__init__()
        raise NotImplementedError("Implement ClassConditioner.__init__")

    def forward(self, h, class_emb):
        raise NotImplementedError("Implement ClassConditioner.forward")
```

## Evaluation settings

One seed (42). The candidate conditioning is trained at all three channel scales (Small, Medium, Large)
for 35,000 steps each, and generated samples are scored by clean-fid against the CIFAR-10 train set
(50,000-image reference statistics). The reported metrics are the best FID over training at each scale —
`best_fid_small`, `best_fid_medium`, `best_fid_large` — lower is better on all three. The improvement
must come from a transferable conditioning design, not from changes to the dataset, labels, loss,
optimizer, sampler, or metric.
