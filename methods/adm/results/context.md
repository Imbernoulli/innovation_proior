## Research question

By mid-2021 diffusion models are an attractive image generator: they are likelihood-based,
trained by a single stationary objective with no adversarial game to balance, they cover the data
distribution well, and they scale cleanly with compute. On CIFAR-10 they already hold the
state of the art. But on large, diverse datasets — ImageNet at 128×128 and 256×256, LSUN — their
Fréchet Inception Distance still trails the best GANs (BigGAN-deep). The concrete goal is to close
that gap on the hard datasets, measured by FID, **without** changing what makes diffusion
attractive — the same epsilon-prediction objective, the same Gaussian noising process, the same
class of samplers. The open question is whether the *denoising network's architecture* is leaving
sample quality on the table: GAN generators and discriminators have been refined for years through
intense architectural search, while the diffusion UNet has been comparatively unexplored on large
data. If a better denoiser architecture exists, it must (a) keep the same input/output contract —
take a noised image and a timestep, return a same-shaped noise prediction — and (b) transfer across
resolution and dataset scale, so a single design choice pays off at every tier.

## Background

A diffusion model generates by reversing a gradual Gaussian noising process. A datapoint x₀ is
noised over T steps into x_T ≈ N(0, I); the model learns to step backward, x_t → x_{t-1}. Ho et al.
(2020) parameterize the reverse step through a noise predictor ε_θ(x_t, t) trained by the simple
mean-squared error ‖ε_θ(x_t, t) − ε‖², with x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε; the reverse mean
μ_θ(x_t, t) is an affine function of ε_θ. Nichol & Dhariwal (2021) extend this by *learning* the
reverse-process variance Σ_θ as an interpolation Σ_θ = exp(v log β_t + (1−v) log β̃_t) between the
two natural bounds, trained with a hybrid objective L_simple + λ L_vlb, which lets the model sample
well with fewer steps; this learned-variance, hybrid-loss recipe is taken as given. Song et al.
(2020, DDIM) give a non-Markovian sampler with the same forward marginals that, with the reverse
noise set to zero, turns any trained ε_θ into a deterministic few-step sampler — the sampler used
in the low-step regime. All of this fixes the *training target* and the *sampler*; what remains
open is the network that computes ε_θ.

The denoiser is a UNet (Ronneberger et al. 2015): a contracting stack of residual blocks and
downsampling convolutions, a symmetric expanding stack with upsampling, and skip connections
joining encoder and decoder feature maps at equal spatial size. The conditioning on the noise
level enters as a timestep embedding projected into each residual block. Two diagnostic facts about
the prevailing diffusion UNet (Ho et al. 2020) frame the problem. First, it places a single global
self-attention layer at exactly one feature resolution, 16×16, with a single head; everything else
is convolutional. Second, it injects the timestep embedding *additively* — the embedding is
projected and added to the residual block's activations. These were reasonable defaults on small
data, but they were never stress-tested on large, diverse images, where long-range structure lives
at more than one scale and the conditioning signal must modulate a much larger, deeper network.
Song et al. (2021, score-SDE) already demonstrated that further UNet changes improve FID on CIFAR-10
and CelebA-64, evidence that the architecture is not yet saturated — but whether such changes carry
to ImageNet-scale data was unestablished.

The architectural toolkit available at the time is mature and off the shelf. Multi-head
self-attention (Vaswani et al. 2017): split the channel embedding into several heads, run an
independent scaled-dot-product attention in each, and concatenate, so the layer can attend to
several distinct relations at once at nearly the cost of one full-width attention; its cost is
O(N²) in the number of spatial positions N = H·W, which is the binding constraint on where it can
be placed in an image network. Group normalization (Wu & He 2018): normalize over groups of
channels within a single example, independent of batch size and therefore stable at the small
batches a large image model is trained with. FiLM (Perez et al. 2018) and adaptive instance
normalization (Karras et al. 2019, StyleGAN): condition a network by having an external vector
produce per-channel *scale and shift* applied to normalized activations, a multiplicative-plus-
additive modulation rather than a bare additive bias. BigGAN (Brock et al. 2019): a residual block
that folds the resolution change into the block itself — the channel-changing convolution also
resamples, with a parallel resampled skip path — giving a learned, residual up/downsampling instead
of a separate fixed pool or interpolation followed by a plain convolution. Residual-branch
rescaling by 1/√2 (Karras et al.; Song et al. 2021): divide each residual sum by √2 to keep the
activation variance from growing as residual branches accumulate down a deep stack. Each of these
is a known building block; none has been systematically combined and ablated inside the diffusion
denoiser on large data.

## Baselines

The prior denoiser designs a new architecture would be measured against and react to.

### The DDPM UNet (Ho et al., 2020)

Core idea: a UNet of stacked residual blocks with downsampling convolutions then upsampling
convolutions and equal-resolution skip connections, computing ε_θ(x_t, t). Self-attention is a
single global layer at the 16×16 feature resolution with one head; the timestep embedding is
projected and *added* into each residual block; downsampling and upsampling are done by separate
convolution/pooling-and-interpolation operations outside the residual blocks. This is the design
that made diffusion competitive on CIFAR-10.

Gap: on large diverse datasets its FID trails GANs. Its global attention sees only one
feature resolution, so coordination of structure at other scales has to be carried indirectly
through the convolutional stack; it uses a single attention head, a narrower relevance pattern than
the multi-head attention standard elsewhere; and the additive timestep injection gives the
conditioning signal only an additive shift over the activations. Whether these specific choices
limit sample quality at ImageNet scale is exactly what is untested.

### The improved-DDPM denoiser (Nichol & Dhariwal, 2021)

Core idea: keep the Ho et al. UNet but learn the reverse-process variance (the v-interpolation
above) and train with the hybrid L_simple + λ L_vlb objective, improving log-likelihood and
few-step sampling. The training target and loss are inherited from here.

Gap: the *architecture* of the denoiser is essentially unchanged from Ho et al. — same
single-resolution single-head attention, same additive conditioning. The likelihood and sampling
improvements do not by themselves bring FID on large data down to GAN levels; the network that
produces ε_θ is still the open variable.

### The score-SDE denoiser (Song et al., 2021)

Core idea: a continuous-time score model whose denoiser is a UNet with several changes relative to
Ho et al. — among them BigGAN-style residual blocks for up/downsampling and 1/√2 residual
rescaling — shown to improve FID on CIFAR-10 and CelebA-64.

Gap: the demonstration is on small, low-resolution, relatively homogeneous datasets. It
establishes that the diffusion UNet is improvable but leaves open which changes matter, in what
combination, and whether the gains transfer to large, diverse, higher-resolution data like
ImageNet — and it does not isolate the contribution of each change.

## Evaluation settings

The natural yardsticks for image generation, all pre-existing. Datasets: ImageNet at 128×128 and
256×256 and LSUN categories for the hard regime; CIFAR-10 (32×32) as the small-data reference.
Pixels are 8-bit, mapped to [−1, 1], with random horizontal flips. The architecture ablations are
run on ImageNet 128×128 with batch size 256 and 250 sampling steps, with FID reported at two points
in training to check that an improvement is not a transient. The
primary metric is FID computed against the full training set as the reference batch (lower is
better), with Inception Score / Precision for fidelity and Recall for diversity as auxiliary
metrics; sFID for spatial structure. Optimization: Adam/AdamW with β₁ = 0.9, β₂ = 0.999, a
parameter EMA at rate 0.9999, mixed-precision (fp16 compute with loss scaling, fp32 weights/EMA/
optimizer state), 1000 diffusion steps. These are the settings into which a candidate architecture
is dropped; none of them is what is being designed.

## Code framework

The denoiser plugs into a fixed diffusion harness: a data pipeline that maps images to [−1, 1] and
batches them; a noising step that forms x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε at a random t; the ε-prediction
MSE loss; a DDPM/DDIM sampler; and an outer loop with AdamW and an EMA. The single editable object
is the network that maps (x_t, t) to a same-shaped noise prediction. Everything below it — residual
block internals, where and how attention is placed, how the timestep embedding is injected, how
resolution changes are done — is the open design space, marked by empty stubs. The scaffold exposes
only generic, pre-existing primitives (a residual block, an optional attention block, a timestep
embedding, the UNet wiring) with the method-specific choices left blank.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    # standard multi-frequency sinusoidal embedding of the scalar noise level
    half = dim // 2
    import math
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    """Time-conditioned residual block. How the embedding is injected, and whether the
    block also changes resolution, are left open."""

    def __init__(self, in_ch, out_ch, emb_ch, dropout=0.0):
        super().__init__()
        # TODO: the block internals (normalization, conditioning mechanism, resampling).
        pass

    def forward(self, x, emb):
        # TODO: return an [N, out_ch, H', W'] tensor conditioned on emb.
        pass


class AttentionBlock(nn.Module):
    """Self-attention over the spatial positions of a feature map. The number of heads
    (or channels per head) is left open."""

    def __init__(self, channels):
        super().__init__()
        # TODO: the attention configuration.
        pass

    def forward(self, x):
        # TODO: return a same-shaped feature map.
        pass


class Denoiser(nn.Module):
    """UNet that maps (x_t, t) -> predicted noise of the same shape. Which resolutions
    carry attention, how resolution changes are implemented, the width/depth/head
    schedule, and the conditioning mechanism are exactly what is to be designed."""

    def __init__(self, in_channels=3, out_channels=3,
                 base_channels=128, channel_mult=(1, 2, 2, 2),
                 num_res_blocks=2, attention_resolutions=(), dropout=0.0):
        super().__init__()
        # TODO: the architecture to design.
        pass

    def forward(self, x, timestep):
        # TODO: return the predicted noise, shape [N, out_channels, H, W].
        pass
```
