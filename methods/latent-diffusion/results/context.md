## Research question

How can we generate high-resolution, photorealistic images of complex natural
scenes — with the broad distribution coverage and stable training of
likelihood-based models — *without* paying the enormous compute cost that the
best such models currently demand?

The sharpest instance of the problem is in denoising diffusion models. They have
become the strongest image generators by sample quality and density estimation,
they avoid the mode collapse and training instability of adversarial methods,
and they support flexible test-time control (inpainting, colorization,
guidance). But they operate directly on the pixel grid and must be evaluated
repeatedly along a long denoising chain. Training a strong model takes on the
order of 150–1000 V100-days; drawing 50k samples takes days on a high-end GPU;
sampling needs tens to a thousand sequential network evaluations. This puts
training out of reach for all but a few labs, leaves a large energy footprint,
and makes even *using* a trained model slow.

A successful method would (a) cut both training and sampling compute by a large
factor, (b) preserve sample quality and the mode-covering behavior of a
likelihood objective, (c) keep the flexible conditioning and guidance that make
diffusion attractive, and (d) scale gracefully to megapixel resolutions.

## Background

**The spatial-cost problem.** The dominant cost of an image generator that works
on pixels is set by the spatial resolution: every layer of a convolutional or
attention backbone does work that grows with the number of spatial positions
(roughly quadratically for the resolution, and worse for global attention).
Diffusion models multiply this by the number of denoising steps and by both the
forward and backward pass during training. The cost of a pixel-space diffusion
model is thus governed jointly by the spatial resolution it processes, the
length of the denoising chain, and the size of the dataset.

**Where the bits go — a rate/distortion observation.** Likelihood-based models
are mode-covering: they must place probability mass on essentially all of the
data, so they spend modeling capacity in proportion to the information content of
the signal. Natural images are dominated, bit-for-bit, by high-frequency texture
that is barely perceptible (this is the same observation that motivated
discretized/quantized likelihoods in PixelCNN++ and the typicality analyses of
likelihood models). If one inspects a *trained* pixel-space diffusion model
through a rate/distortion lens, learning splits into two regimes. A first regime
spends a large share of the bits on **perceptual compression** — removing
imperceptible high-frequency detail — while changing almost nothing about the
semantics or layout of the image. A second regime, with the remaining bits, does
the actual **semantic/conceptual** modeling — what objects are present and how
they are arranged. The same expensive sequential network is run at full
pixel-space resolution across both regimes.

**The denoising-diffusion machinery.** A diffusion model defines a fixed forward
process that gradually adds Gaussian noise. With signal/noise schedules
$(\alpha_t),(\sigma_t)$ and signal-to-noise ratio
$\mathrm{SNR}(t)=\alpha_t^2/\sigma_t^2$,

$$q(x_t\mid x_0)=\mathcal N(x_t\mid \alpha_t x_0,\ \sigma_t^2 I),$$

with a Markov structure $q(x_t\mid x_s)=\mathcal N(\alpha_{t\mid s}x_s,\ \sigma_{t\mid s}^2 I)$ for $s<t$, where $\alpha_{t\mid s}=\alpha_t/\alpha_s$ and $\sigma_{t\mid s}^2=\sigma_t^2-\alpha_{t\mid s}^2\sigma_s^2$. A generative model reverses it, $p(x_0)=\int p(x_T)\prod_t p(x_{t-1}\mid x_t)$, with $p(x_T)$ a standard normal. The model is trained against the evidence lower bound, which decomposes over time steps into KL terms between the true posterior $q(x_{t-1}\mid x_t,x_0)$ and the learned reverse step. Parameterizing the reverse step through the true posterior with the unknown clean signal replaced by a network estimate, and reparameterizing to predict the injected noise, collapses the bound into a sum of weighted denoising regressions; assigning every step equal weight yields the simple, stable objective that the strongest models use.

**Spatial inductive bias.** Diffusion models owe much of their image quality to a
UNet backbone: a convolutional encoder/decoder with skip connections that matches
the local, translation-equivariant structure of images and conditions on the
timestep. Any method that keeps a 2D, grid-structured representation can keep
this inductive bias; methods that flatten the representation into a 1D sequence
cannot.

**Conditioning and guidance.** Diffusion models can model conditional
distributions $p(x\mid y)$, and they support test-time control. A trained
unconditional model can be steered by a separately trained noise-aware
classifier $\log p_\Phi(y\mid x_t)$ whose gradient nudges the denoising update.
A classifier-free variant trains a single network with the conditioning randomly
dropped and forms a guided prediction by extrapolating between the conditional
and unconditional outputs, trading diversity for fidelity without a separate
classifier.

## Baselines

**GANs** (Goodfellow et al. 2014; BigGAN, Brock et al. 2019; StyleGAN, Karras
et al. 2019). A generator and discriminator play a minimax game. They sample a
high-resolution image in a single forward pass and produce sharp results, but
adversarial training is hard to optimize and prone to mode collapse, so they
miss parts of the data distribution (low recall). Gap: distribution coverage and
training stability.

**VAEs and normalizing flows** (Kingma & Welling 2013; RealNVP/Glow, Dinh et al.
2017). Likelihood-based, well-behaved optimization, fast sampling. A VAE encodes
to a Gaussian latent with a KL term toward $\mathcal N(0,1)$ and decodes; a flow
uses invertible maps for exact likelihood. Gap: sample quality lags GANs.

**Autoregressive models** (PixelRNN/PixelCNN, van den Oord et al. 2016; PixelCNN++,
Salimans et al. 2017). Factorize the image likelihood pixel-by-pixel and achieve
strong density estimation, but sampling is inherently sequential over all
positions, so they are confined to low resolution. Gap: sampling cost scales
catastrophically with resolution.

**Denoising diffusion models** (Sohl-Dickstein et al. 2015; DDPM, Ho et al.
2020; score-based SDEs, Song et al. 2021). The forward/reverse process above with
an $\epsilon$-prediction UNet trained on the simplified objective
$\mathbb E_{x,\epsilon,t}\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2$. SOTA quality
and coverage, flexible. Gap: every step is a full pixel-space network
evaluation, so training (hundreds of GPU-days) and sampling (many sequential
steps) are extremely expensive.

**Class-conditional pixel diffusion** (ADM / "Diffusion Models Beat GANs",
Dhariwal & Nichol 2021). The strongest pixel-space diffusion model at the time,
with an "ablated UNet" (BigGAN-style residual blocks, attention at several
resolutions) and classifier guidance. It is precisely the model whose 150–1000
V100-day training cost defines the pain. Gap: cost.

**Two-stage compress-then-model** (VQ-VAE, van den Oord et al. 2017; VQ-VAE-2,
Razavi et al. 2019; VQGAN / Taming Transformers, Esser et al. 2020; DALL·E,
Ramesh et al. 2021). First learn a discrete latent with an autoencoder — for
VQGAN, an encoder/decoder trained with a perceptual loss plus a patch-based
adversarial loss and a vector-quantization codebook (perceptual + adversarial
losses keep reconstructions on the image manifold and avoid the blur of pure
$L_2$/$L_1$). Then model the prior over the latent codes with an autoregressive
transformer. This is the right decomposition, but the second-stage transformer's
cost is quadratic in sequence length, so it forces a *high* compression rate
(e.g. 16×) to keep the token sequence short. That discards detail, requires
billions of transformer parameters, and serializes the 2D latent into a 1D
raster order that ignores its spatial structure. Gap: the compression level is
dictated by the transformer's appetite, not by what preserves quality.

**Jointly learned latent + score prior** (LSGM, Vahdat et al. 2021). Learns the
encoder/decoder *and* a score-based diffusion prior over the latent
simultaneously. Gap: jointly optimizing reconstruction and the generative prior
requires a delicate weighting between the two, and the latent space is a moving
target while the prior is being learned.

## Evaluation settings

Datasets in standard use: ImageNet (class-conditional, 1000 classes);
CelebA-HQ and FFHQ (faces); LSUN Churches and Bedrooms; MS-COCO and LAION-400M
(text–image pairs); OpenImages and COCO (bounding-box / layout and segmentation
annotations); Places (inpainting); plus bicubic-downsampled ImageNet for
super-resolution following the SR3 protocol.

Metrics: Fréchet Inception Distance (FID) for sample quality; Inception Score
(IS); Precision and Recall for the fidelity/coverage split; PSNR and SSIM for
super-resolution (with the caveat that they favor blur over imperfectly aligned
high-frequency detail and align poorly with human perception); LPIPS for
perceptual similarity; and two-alternative forced-choice human preference
studies for super-resolution and inpainting. Sample statistics are typically
estimated from 50k generated samples against the full training set, using
established FID/Precision/Recall implementations.

Compute is reported in GPU-days (V100-equivalent), and sampling speed in
samples/second versus FID at a given number of denoising steps, so that quality
and cost can be read off together.

## Code framework

The pieces that already exist: a `pytorch_lightning` training loop, Adam, a
time-conditional image backbone with timestep embeddings, denoising-diffusion
noise schedules with a closed-form forward sampler, and the building blocks of
an autoencoding reconstruction setup (perceptual feature distance, a patchwise
discriminator). The remaining slot is the generative model itself, left empty
for the method to fill.

```python
import torch
import torch.nn as nn

# --- existing primitives -------------------------------------------------

class ImageBackbone(nn.Module):
    """Existing time-conditional image backbone abstraction."""
    def forward(self, x, t, conditioning=None):
        pass

def make_noise_schedule(num_steps, start=1e-4, end=2e-2):
    # TODO: return the existing scalar schedule buffers.
    pass

def add_noise(clean, t, noise, signal_scale, noise_scale):
    # closed-form forward sampler for an existing denoising process
    signal = signal_scale[t].view(-1, *([1] * (clean.ndim - 1)))
    sigma = noise_scale[t].view(-1, *([1] * (clean.ndim - 1)))
    return signal * clean + sigma * noise

class PerceptualDistance(nn.Module):
    def forward(self, x, x_rec):
        pass

class PatchDiscriminator(nn.Module):
    def forward(self, x):
        pass


# --- slot the method will fill ------------------------------------------

class Generator(nn.Module):
    """The generative model. TODO."""
    def __init__(self, backbone: ImageBackbone):
        super().__init__()
        self.backbone = backbone

    def training_objective(self, x, y=None):
        # TODO
        pass

    @torch.no_grad()
    def sample(self, shape, y=None):
        # TODO: produce an image
        pass
```
