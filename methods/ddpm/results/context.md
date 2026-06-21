## Research question

The concrete goal, as of mid-2020, is a generative model of natural images that is **likelihood-based and non-adversarial**: a model trained by minimizing a single scalar objective (a bound on negative log-likelihood) with ordinary SGD, no min-max game and no discriminator, that nonetheless produces sharp, globally coherent samples on CIFAR10 (32×32) and on 256×256 faces and scenes. How should such a model be specified and trained so that it reaches sample quality competitive with the best generators while keeping a principled, measurable training signal?

## Background

The field, mid-2020, has several mature toolkits whose pieces are individually well understood:

- **Variational inference for latent-variable models.** For p(x)=∫p(x,z)dz with posterior q(z|x), Jensen gives the ELBO E_q[log p(x,z) − log q(z|x)] ≤ log p(x), training decoder and inference model jointly. The **reparameterization trick** (z=μ+σ⊙ε, ε fixed-noise) makes the sampling step a differentiable function of the parameters, giving low-variance pathwise gradients for SGD.
- **Closed-form KL.** When the two distributions compared in an objective are both Gaussian, the KL (or cross-entropy) term has an analytic form; replacing a Monte-Carlo estimate of it with the closed form removes that term's variance.
- **Score-based modeling.** Instead of the density one can model the **score** ∇_x log p(x); for a Gaussian-perturbed density the score is estimable by regression (no normalizing constant), and **Langevin dynamics** x ← x + (δ/2)∇ log p(x) + √δ·z turns a score function into a sampler.
- **The denoising ↔ score identity.** For a Gaussian corruption kernel, learning to denoise is, up to a constant, the same as learning the score of the corrupted density.
- **Strong image-to-image conditional CNNs.** U-Net encoder–decoders with skip connections, residual blocks, group normalization, sinusoidal scalar-conditioning embeddings, and self-attention at coarse resolutions are off-the-shelf backbones for dense image prediction.
- **Discretized continuous likelihoods.** Integrating a continuous density over per-pixel bins yields a proper lossless codelength on 8-bit data without dequantization noise. After mapping 0…255 to [-1,1], adjacent values are spaced 2/255 apart, so each interior bin has half-width 1/255.

## Baselines

### Sohl-Dickstein, Weiss, Maheswaranathan & Ganguli (2015) — generative modeling via nonequilibrium thermodynamics

Core idea: define a **fixed** forward Markov chain q that, over many steps, slowly adds noise to a datapoint until all structure is destroyed and the endpoint is a simple known distribution. Then **learn** a reverse Markov chain p_θ that undoes the corruption step by step; running it from the simple endpoint generates data. The structural fact it relies on is that if each forward step adds only a tiny amount of Gaussian noise, the *reverse* conditional of each step is also approximately Gaussian — so a model whose reverse transitions are Gaussian is expressive enough to invert the process. Training maximizes a variational bound on the data log-likelihood, exactly as in a deep latent-variable model, but the inference distribution (the forward chain) is fixed and has **no parameters**.

Algorithm/math: a forward chain q(x_t|x_{t-1}) of Gaussian noising steps, a learned reverse chain p_θ(x_{t-1}|x_t) of Gaussian denoising steps, and a tractable variational bound obtained by comparing them term by term.

### Kingma & Welling (2013) — the variational autoencoder

Core idea: a latent-variable generative model p_θ(x|z)p(z) trained with an amortized inference network q_φ(z|x) by maximizing the ELBO log p(x) ≥ E_q[log p_θ(x|z)] − KL(q_φ(z|x)‖p(z)). The reparameterization trick z=μ_φ(x)+σ_φ(x)⊙ε makes the bound differentiable end-to-end. The machinery it contributes is the ELBO and reparameterization for a single-step amortized posterior.

### Song & Ermon (2019) — Noise Conditional Score Networks (NCSN) + annealed Langevin

Core idea: estimate the score ∇_x log p_σ(x) of the data perturbed by Gaussian noise at a geometric ladder of scales σ_1 > σ_2 > … > σ_L, using **one** noise-conditional network s_θ(x,σ) trained by denoising score matching: Σ_σ λ(σ)·E_{x,x̃}‖s_θ(x̃,σ) − ∇_{x̃} log q_σ(x̃|x)‖². Sample by **annealed Langevin dynamics**: run Langevin steps at the largest σ, then anneal σ down, so the chain flows from noise toward data.

Algorithm/math: per-scale denoising regression for training; a hand-set sequence of Langevin step sizes and per-scale noise injections for sampling. The perturbation adds noise without rescaling the data.

### Vincent (2011) — the connection between score matching and denoising autoencoders

Core idea: for a Gaussian corruption q_σ(x̃|x)=N(x̃; x, σ²I), minimizing the denoising-score-matching objective E‖s_θ(x̃) − ∇_{x̃} log q_σ(x̃|x)‖² equals explicit score matching against the perturbed density, up to a constant. Here ∇_{x̃} log N(x̃; x, σ²I) = (x − x̃)/σ², which is proportional to the negative of the noise that was added. This is an identity relating a denoising regression to score estimation for a single fixed corruption scale.

### Langevin dynamics (the background sampler)

Given a score function, the update x ← x + (δ/2)∇ log p(x) + √δ·z (z standard normal) is a discretized Langevin diffusion whose stationary distribution is p. NCSN uses it as its sampler. Its characteristic shape is "move along a gradient, then add a little fresh noise".

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

def get_step_variance_schedule(schedule, *, start, end, steps):
    # TODO: choose the latent path's step-variance schedule.
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
    """Same-shape image network with optional scalar conditioning."""
    def __init__(self, image_size=32, channels=3, base_channels=128,
                 out_channels=None, channel_mult=(1, 2, 4, 8),
                 num_res_blocks=2, attn_resolutions=(), dropout=0.0):
        super().__init__()
        # TODO: choose the architecture and conditioning mechanism.
        pass

    def forward(self, x, t):
        # TODO: return an image-shaped tensor.
        pass

class ImageLatentGenerator(nn.Module):
    """Latent-variable image generator trained by a scalar objective."""
    def __init__(self, backbone, image_size, latent_steps=1000,
                 schedule="linear", variance_start=None, variance_end=None,
                 variance_type=None,
                 prediction_type=None, loss_type=None):
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
        # TODO: scalar loss for a training step.
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
