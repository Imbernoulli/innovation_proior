# Context: likelihood-based image generation, mid-2020

## Research question

The concrete goal is an unconditional generative model of natural images — think 32×32 CIFAR10
samples — that hits four targets at once which, as of mid-2020, no single model family delivers
together:

1. **Sample quality** on par with the best adversarial models: sharp, globally coherent images,
   measured by Inception Score and Fréchet Inception Distance against the training set.
2. **Stable, non-adversarial training** — a single scalar objective that ordinary stochastic
   gradient descent minimizes, with no min-max game, no discriminator to balance, no mode
   collapse to nurse.
3. **Likelihood-based** — the model optimizes a proper bound on negative log-likelihood, giving
   a principled training signal and a measurable codelength in bits per dimension, not only an
   implicit sample-quality proxy.
4. **A simple, parameter-light definition** — easy to specify and scale, ideally with an
   inference path that carries no parameters to overfit or collapse.

The pain is that the mature families each purchase one or two of these at the cost of another.
Adversarial models reach (1) but sacrifice (2) and (3). Autoregressive and flow-based models
secure (3) but pay in sampling cost or expressiveness. Latent-variable models with learned
inference networks secure (2) and (3) but historically not (1) — they produce blurry samples.
A solution would have to deliver high-fidelity samples from a likelihood-based, non-adversarial
model that is cheap to define and train.

## Background

The field, mid-2020, has several mature toolkits whose pieces are individually well understood.

- **Variational inference for latent-variable models.** For a marginal p(x)=∫p(x,z)dz with an
  approximate posterior q(z|x), Jensen's inequality gives the evidence lower bound
  E_q[log p(x,z) − log q(z|x)] ≤ log p(x); maximizing it jointly fits a generative decoder and an
  inference model. The **reparameterization trick** writes a sampling step z=μ+σ⊙ε with ε a
  fixed-noise variable, turning the stochastic draw into a differentiable function of the
  parameters and yielding low-variance pathwise gradients usable with SGD (Kingma & Welling 2013).
- **Rao-Blackwellization / closed-form KL.** When two distributions compared inside an objective
  are both Gaussian, the KL divergence (and the cross-entropy term) has an analytic form. Swapping
  a Monte-Carlo estimate of such a term for its closed form removes that term's sampling variance
  — a standard variance-reduction move.
- **Score-based modeling.** Instead of the density one can model the **score** ∇_x log p(x). For a
  Gaussian-perturbed density the score is estimable by regression, with no normalizing constant,
  and **Langevin dynamics** — x ← x + (δ/2)∇ log p(x) + √δ·z, z standard normal — turns a score
  function into a sampler whose stationary distribution is p.
- **The denoising ↔ score identity.** For a Gaussian corruption kernel, learning to denoise is, up
  to a constant, the same as learning the score of the corrupted density (Vincent 2011).
- **Strong image-to-image conditional CNNs.** Encoder–decoder networks with skip connections,
  residual blocks, group normalization (Wu & He 2018), sinusoidal embeddings for a scalar
  conditioning input (Vaswani et al. 2017), and self-attention at coarse feature resolutions are
  off-the-shelf and known to be effective backbones for dense image prediction; the PixelCNN++
  backbone (Salimans et al. 2017), a U-Net (Ronneberger et al. 2015) over Wide-ResNet blocks
  (Zagoruyko & Komodakis 2016), is a standard such design.
- **Discretized continuous likelihoods.** Treating 8-bit pixel values as bins and integrating a
  continuous density over each bin yields a proper lossless codelength on discrete data without
  dequantization noise. After mapping the integer values 0 through 255 to [−1,1], adjacent values
  are spaced 2/255 apart, so each interior bin has half-width 1/255.

The prevailing wisdom is that high-fidelity image synthesis requires either adversarial training
or expensive autoregressive decoding, and that likelihood-based latent-variable models are doomed
to blurry samples. A diagnostic fact about the score-based road sharpens the picture: a perturbed
process that adds noise *without* rescaling the signal lets the variance of the perturbed data
grow with the noise scale, so a single network sees inputs at wildly inconsistent scales across
noise levels; and when the largest perturbation does not drive the data all the way to a fixed,
known endpoint distribution, the sampler is started from a distribution that the network was never
trained to expect.

## Baselines

The prior methods an image generator would be measured against, and react to.

### Generative modeling by a fixed-corruption / learned-reversal Markov chain (Sohl-Dickstein, Weiss, Maheswaranathan & Ganguli, 2015)

Core idea: define a **fixed** forward Markov chain q that, over many steps, slowly adds noise to a
datapoint until all structure is destroyed and the endpoint becomes a simple known distribution.
Then **learn** a reverse Markov chain p_θ that undoes the corruption step by step; running it from
the simple endpoint generates data. The structural fact that makes it work: if each forward step
adds only a tiny amount of Gaussian noise, the *reverse* conditional of that step is also
approximately Gaussian, so a model whose reverse transitions are Gaussian is expressive enough to
invert the chain. Training maximizes a variational bound on the data log-likelihood, exactly as in
a deep latent-variable model, but the inference distribution (the forward chain) is fixed and has
**no parameters** — no encoder to collapse, no posterior to learn.

Algorithm/math: a forward chain q(x_t|x_{t-1}) of Gaussian noising steps, a learned reverse chain
p_θ(x_{t-1}|x_t) of Gaussian denoising steps, and a tractable variational bound obtained by
comparing the two chains term by term.

Gap: as instantiated, this framework was not a high-fidelity image generator — on CIFAR10 its
codelength reached only about 5.40 bits per dimension and its samples were visually weak. Many of
the choices that would turn the abstract framework into a competitive image model are left
unspecified: the parameterization of the reverse mean, the reverse variances, the noise schedule,
the network architecture, and the weighting of the objective terms.

### The variational autoencoder (Kingma & Welling, 2013)

Core idea: a latent-variable generative model p_θ(x|z)p(z) trained with an amortized inference
network q_φ(z|x) by maximizing the ELBO log p(x) ≥ E_q[log p_θ(x|z)] − KL(q_φ(z|x)‖p(z)). The
reparameterization trick z=μ_φ(x)+σ_φ(x)⊙ε makes the bound differentiable end to end and supplies
the low-variance gradients that make training practical.

Gap: a single-step amortized posterior tends to **collapse** — the decoder learns to ignore the
latent — and the samples come out **blurry**, because one Gaussian latent layer is a short bridge
between a complex data distribution and a simple prior. A deep stack of latent layers helps but is
hard to train. What this contributes is the machinery — the ELBO and reparameterization — not a
strong image model.

### Noise-conditional score networks with annealed Langevin sampling (Song & Ermon, 2019)

Core idea: estimate the score ∇_x log p_σ(x) of the data perturbed by Gaussian noise at a
geometric ladder of scales σ_1 > σ_2 > … > σ_L, with **one** noise-conditional network s_θ(x,σ)
trained by denoising score matching, Σ_σ λ(σ)·E_{x,x̃}‖s_θ(x̃,σ) − ∇_{x̃} log q_σ(x̃|x)‖². Sample
by **annealed Langevin dynamics**: run Langevin steps at the largest σ, then anneal σ down so the
chain flows from noise toward data. Sample quality approaches the adversarial models'.

Algorithm/math: per-scale denoising regression for training; a hand-set sequence of Langevin step
sizes and per-scale noise injections for sampling.

Gaps: (1) the noise schedule and the **sampler coefficients** — step sizes, per-scale noise
amounts — are tuned **by hand, after the fact**; the training objective does not directly optimize
the sampler. (2) The perturbation does **not rescale** the data, so the variance of the noised
data grows with σ and presents inconsistently scaled inputs to the network. (3) The largest-σ
perturbation does **not** drive the data to a fixed known endpoint, so the sampler's start
distribution does not match the trained noise levels. (4) It is **not** a likelihood-based
latent-variable model — no variational bound, no codelength, and no guarantee the sampler
optimizes a quality metric of the model.

### The denoising ↔ score-matching identity (Vincent, 2011)

Core idea: for a Gaussian corruption q_σ(x̃|x)=N(x̃; x, σ²I), minimizing the denoising
score-matching objective E‖s_θ(x̃) − ∇_{x̃} log q_σ(x̃|x)‖² equals explicit score matching against
the perturbed density, up to a constant. The gradient ∇_{x̃} log N(x̃; x, σ²I) = (x − x̃)/σ² is
proportional to the negative of the noise that was added, so **regressing onto the added noise is
regressing onto the score**, up to a positive scale.

Gap: this is an identity, not a model. It relates a denoising regression to score estimation at a
single fixed corruption scale; it says nothing about chaining noise levels into a generator.

### Langevin dynamics (the background sampler)

Given a score function, the update x ← x + (δ/2)∇ log p(x) + √δ·z is a discretized Langevin
diffusion whose stationary distribution is p. Its characteristic shape is "move along a gradient,
then add a little fresh noise." It is the sampler the score-based road relies on.

## Evaluation settings

The natural yardsticks already in place for unconditional image generation:

- **Datasets.** CIFAR10 (32×32, unconditional); for higher resolution, CelebA-HQ and LSUN
  categories at 256×256, loaded by standard splits. Pixels are 8-bit integers per channel.
- **Sample-quality metrics.** Inception Score and Fréchet Inception Distance, computed on
  50,000 samples against the training set, using the established evaluation code; lower FID is
  better.
- **Likelihood metric.** Negative log-likelihood / lossless codelength in bits per dimension on
  held-out data, the standard for likelihood-based models; the train/test gap is the overfitting
  check.
- **Protocol.** Map 8-bit pixels to [−1,1]; standard augmentation (random horizontal flips). Train
  once with a first-order optimizer (Adam-class) and a parameter exponential moving average; track
  sample-quality metrics throughout and report at the best checkpoint.

## Code framework

The model plugs into a generic latent-variable image-generation harness: a data pipeline that maps
images to [−1,1] and batches them, a same-shape image network with an optional scalar conditioning
input, an object that defines a latent generative process and exposes a scalar training loss and a
sampler, and an outer loop over batches with a first-order optimizer and a parameter EMA. Nothing
about the latent process, the loss, or the sampler is settled — that is exactly what is to be
designed — so the substrate is only the generic machinery, with empty slots where the contribution
will go.

```python
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms as T


def get_step_variance_schedule(schedule, *, start, end, steps):
    # TODO: choose the latent path's per-step variance schedule.
    pass


def extract(coefs, t, x_shape):
    # TODO: gather per-step coefficients and broadcast them to image tensors.
    pass


def to_model_input(img):
    return img * 2.0 - 1.0


transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Lambda(to_model_input),
])


class ImageBackbone(nn.Module):
    """Same-shape image network with an optional scalar conditioning input."""

    def __init__(self, image_size=32, channels=3, base_channels=128,
                 out_channels=None, channel_mult=(1, 2, 2, 2),
                 num_res_blocks=2, attn_resolutions=(), dropout=0.0):
        super().__init__()
        # TODO: choose the architecture and the conditioning mechanism.
        pass

    def forward(self, x, t):
        # TODO: return an image-shaped tensor.
        pass


class ImageLatentGenerator(nn.Module):
    """Latent-variable image generator trained by a scalar objective."""

    def __init__(self, backbone, image_size, latent_steps=1000,
                 schedule="linear", variance_start=None, variance_end=None,
                 variance_type=None, prediction_type=None, loss_type=None):
        super().__init__()
        self.backbone = backbone
        self.image_size = image_size
        self.latent_steps = latent_steps
        # TODO: define the latent process and what the generator is trained to do.
        pass

    def training_losses(self, x0, t=None, noise=None):
        # TODO: per-example losses for fitting the generator to data x0.
        pass

    def training_loss(self, x0, t=None, noise=None):
        # TODO: scalar loss for one training step.
        pass

    @torch.no_grad()
    def sample(self, batch_size, device=None):
        # TODO: draw a batch of images from the trained generator.
        pass


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        online = model.state_dict()
        for name, value in self.ema_model.state_dict().items():
            src = online[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                value.copy_(src)


model = ImageLatentGenerator(ImageBackbone(image_size=32), image_size=32)
opt = Adam(model.parameters(), lr=2e-4)
ema = ModelEMA(model, decay=0.9999)

for x0 in dataloader:
    loss = model.training_loss(x0)
    opt.zero_grad()
    loss.backward()
    opt.step()
    ema.update(model)
```
