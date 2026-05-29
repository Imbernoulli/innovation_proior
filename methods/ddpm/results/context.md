# Context: the ground a high-fidelity, likelihood-based image generator would stand on (mid-2020)

## Research question

The concrete goal is a generative model of natural images that simultaneously hits four targets that, as of mid-2020, no single family achieves together:

1. **Sample quality** competitive with the best adversarial models (sharp, globally coherent CIFAR10 / 256×256 faces and scenes).
2. **Stable, non-adversarial training** — a single scalar objective minimized by ordinary SGD, with no min-max game, no discriminator balancing, no mode collapse to manage.
3. **Likelihood-based** — the model should optimize (a bound on) negative log-likelihood, so it has a principled training signal and a measurable codelength, rather than only an implicit sample-quality proxy.
4. **A simple, parameter-light definition** — easy to specify, easy to scale, ideally with an inference path that has no parameters to overfit or collapse.

The pain point is that the existing families each buy one or two of these at the cost of another: adversarial models get (1) but not (2) or (3); autoregressive and flow models get (3) but pay in sampling cost or expressiveness; latent-variable models with learned encoders get (2)/(3) but historically not (1). A solution would have to deliver high-quality samples from a likelihood-based, non-adversarial model that is cheap to define and train.

## Background

The field, mid-2020, has several mature toolkits whose pieces are individually well understood:

- **Variational inference for latent-variable models.** For p(x)=∫p(x,z)dz with an approximate posterior q(z|x), Jensen gives the evidence lower bound E_q[log p(x,z) − log q(z|x)] ≤ log p(x). Maximizing it trains a generative decoder and an inference model jointly. The **reparameterization trick** (sampling z=μ+σ⊙ε with ε a fixed-noise variable) turns the sampling step into a differentiable function of the parameters, giving low-variance pathwise gradients usable with SGD.
- **Rao-Blackwellization / closed-form KL.** When two distributions being compared in an objective are both Gaussian, the KL (or the cross-entropy term) has an analytic form. Replacing a Monte-Carlo estimate of such a term with its closed form removes that term's variance — a standard variance-reduction move.
- **Score-based modeling.** Instead of the density, one can model the **score** ∇_x log p(x). For a Gaussian-perturbed density, the score is estimable by regression (no normalizing constant needed), and **Langevin dynamics** — x ← x + (δ/2)∇ log p(x) + √δ·z — turns a score function into a sampler.
- **The denoising ↔ score identity.** For a Gaussian corruption kernel, learning to denoise is, up to a constant, the same as learning the score of the corrupted density. This is the hinge that lets "predict the noise that was added" stand in for "estimate the gradient of log-density".
- **Strong image-to-image conditional CNNs.** U-Net encoder–decoders with skip connections, residual blocks, group normalization, sinusoidal scalar-conditioning embeddings, and self-attention at coarse resolutions are off-the-shelf and known to be effective backbones for dense image prediction.
- **Discretized continuous likelihoods.** Treating 8-bit pixel values as bins and integrating a continuous density over each bin yields a proper lossless codelength on discrete data without dequantization noise. After mapping integer values 0 through 255 to [-1,1], adjacent values are spaced 2/255 apart, so each interior bin has half-width 1/255.

The prevailing wisdom is that high-quality image synthesis requires either adversarial training or expensive autoregressive decoding; likelihood-based latent-variable models are assumed to produce blurry samples.

## Baselines

### Sohl-Dickstein, Weiss, Maheswaranathan & Ganguli (2015) — generative modeling via nonequilibrium thermodynamics

Core idea: define a **fixed** forward Markov chain q that, over many steps, slowly adds noise to a datapoint until all structure is destroyed and the endpoint is a simple known distribution. Then **learn** a reverse Markov chain p_θ that undoes the corruption step by step; running it from the simple endpoint generates data. The crucial structural fact is that if each forward step adds only a tiny amount of Gaussian noise, the *reverse* conditional of each step is also approximately Gaussian — so a model whose reverse transitions are Gaussian is expressive enough to invert the process. Training maximizes a variational bound on the data log-likelihood, exactly as in a deep latent-variable model, but the inference distribution (the forward chain) is fixed and has **no parameters** — so there is no encoder to collapse and no posterior to learn.

Algorithm/math: a forward chain q(x_t|x_{t-1}) of Gaussian noising steps, a learned reverse chain p_θ(x_{t-1}|x_t) of Gaussian denoising steps, and a tractable variational bound obtained by comparing them term by term.

Gap: this framework was not yet a high-fidelity image generator. What is left undetermined is the *parameterization* of the reverse mean, the choice of reverse variances, the noise schedule, the network architecture, and the training objective weighting — all the choices that turn the abstract framework into a practical image model.

### Kingma & Welling (2013) — the variational autoencoder

Core idea: a latent-variable generative model p_θ(x|z)p(z) trained with an amortized inference network q_φ(z|x) by maximizing the ELBO log p(x) ≥ E_q[log p_θ(x|z)] − KL(q_φ(z|x)‖p(z)). The reparameterization trick z=μ_φ(x)+σ_φ(x)⊙ε makes the bound differentiable end-to-end.

Gap: a single-step amortized posterior tends to **collapse** (latents ignored) and produces **blurry** samples, because one Gaussian latent layer is a weak bridge between a complex data distribution and a simple prior. A deep stack of latent layers helps but is hard to train. The relevant lesson it contributes is the machinery — the ELBO and reparameterization — not a strong image model.

### Song & Ermon (2019) — Noise Conditional Score Networks (NCSN) + annealed Langevin

Core idea: estimate the score ∇_x log p_σ(x) of the data perturbed by Gaussian noise at a geometric ladder of scales σ_1 > σ_2 > … > σ_L, using **one** noise-conditional network s_θ(x,σ) trained by denoising score matching: Σ_σ λ(σ)·E_{x,x̃}‖s_θ(x̃,σ) − ∇_{x̃} log q_σ(x̃|x)‖². Sample by **annealed Langevin dynamics**: run Langevin steps at the largest σ, then anneal σ down, so the chain flows from noise toward data.

Algorithm/math: per-scale denoising regression for training; a hand-set sequence of Langevin step sizes and per-scale noise injections for sampling.

Gaps: (1) the noise schedule and the **sampler coefficients** (step sizes, per-scale noise) are tuned **by hand, post-hoc** — training does not directly optimize the sampler. (2) The perturbation does **not rescale** the data, so the variance of the noised data grows with σ, presenting inconsistently scaled inputs to the network. (3) The largest-σ perturbation does **not** drive the data to a fixed known prior — the forward corruption does not truly "destroy signal" to a standard normal, so there is a mismatch between the sampler's start distribution and the trained noise levels. (4) It is **not** a likelihood-based latent-variable model — no variational bound, no codelength, and no guarantee the sampler optimizes a quality metric of the model.

### Vincent (2011) — the connection between score matching and denoising autoencoders

Core idea: for a Gaussian corruption q_σ(x̃|x)=N(x̃; x, σ²I), minimizing the denoising-score-matching objective E‖s_θ(x̃) − ∇_{x̃} log q_σ(x̃|x)‖² equals explicit score matching against the perturbed density, up to a constant. Crucially, ∇_{x̃} log N(x̃; x, σ²I) = (x − x̃)/σ², which is proportional to the negative of the noise that was added. So **regressing onto the added noise is regressing onto the score**, up to a positive scale.

Gap: this is an identity, not a model — it supplies the bridge that will let a "predict the noise" objective be read as score estimation, but it says nothing about how to chain noise levels into a generator.

### Langevin dynamics (the background sampler)

Given a score function, the update x ← x + (δ/2)∇ log p(x) + √δ·z (z standard normal) is a discretized Langevin diffusion whose stationary distribution is p. NCSN uses it as its sampler. Any reverse-denoising step that takes the shape "move along a learned gradient, then add a little fresh noise" is structurally a Langevin step — a fact that will matter when a denoising reverse chain is compared against this sampler.

## Evaluation settings

The natural yardsticks already in place for unconditional image generation:

- **Datasets.** CIFAR10 (32×32, unconditional), CelebA-HQ (256×256), and LSUN categories (Bedroom, Church, Cat) at 256×256, loaded by standard splits; pixels are 8-bit integers per channel.
- **Sample-quality metrics.** Inception Score (IS) and Fréchet Inception Distance (FID), computed on 50k samples against the training set (the standard practice), using the established OpenAI/TTUR/StyleGAN2 evaluation code.
- **Likelihood metric.** Negative log-likelihood / lossless codelength in **bits per dimension** on held-out data, the standard for likelihood-based models; a small train/test gap is the overfitting check.
- **Protocol.** Train once; track sample-quality metrics throughout training; report at the best checkpoint. Standard augmentations (random horizontal flips) and standard preprocessing (8-bit pixels mapped to [−1,1]).

## Code framework

```python
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms as T

def to_model_input(img):
    return img * 2.0 - 1.0

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Lambda(to_model_input),
])

class ImageBackbone(nn.Module):
    """Same-shape image network with optional scalar conditioning."""
    def __init__(self, channels=3, cond_dim=None, base_channels=128):
        super().__init__()
        # TODO: choose the architecture and conditioning mechanism.
        pass

    def forward(self, x, cond=None):
        # TODO: return an image-shaped tensor.
        pass

class ImageLatentGenerator(nn.Module):
    """Latent-variable image generator trained by a scalar objective."""
    def __init__(self, backbone, image_size, latent_steps=None):
        super().__init__()
        self.backbone = backbone
        self.image_size = image_size
        self.latent_steps = latent_steps
        # TODO: choose the latent process and cache any fixed coefficients.
        pass

    def training_loss(self, x0):
        # TODO: scalar loss for fitting the generator to data x0.
        pass

    @torch.no_grad()
    def sample(self, batch_size):
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

model = ImageLatentGenerator(ImageBackbone(), image_size=32)
opt = Adam(model.parameters(), lr=2e-4)
ema = ModelEMA(model, decay=0.9999)

for x0 in dataloader:
    loss = model.training_loss(x0)
    opt.zero_grad()
    loss.backward()
    opt.step()
    ema.update(model)
```
