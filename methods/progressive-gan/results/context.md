# Context: training adversarial image generators at high resolution

## Research question

Adversarial generative models produce sharp images, but only at fairly small resolutions (32²–256²) and with limited variation, and even there the training is fragile. The goal is to generate high-resolution images — up to 1024×1024 — that are both convincing in fine detail and varied across the dataset, with a training procedure that does not collapse or stall. Reaching megapixel output requires a method that survives at high resolution rather than merely scaling up an existing low-resolution recipe.

Two facts make this hard. First, the gradient the generator receives from the discriminator becomes useless precisely when the two image distributions are easy to tell apart, and *higher resolution makes them easier to tell apart* — more pixels expose more telltale statistics, so the discriminator becomes near-perfect faster and the gradient it hands back points in nearly random directions. Generating directly at 1024² is therefore the worst case for this failure mode. Second, memory limits force smaller minibatches at high resolution, which further degrades the stability of training. On top of this, adversarial generators tend to capture only a subset of the variation in the data (mode collapse), and the two networks can enter an escalation where signal magnitudes spiral out of control.

A solution would have to: make the optimization tractable at high resolution; supply the generator with a usable gradient throughout training; actively encourage variation; and prevent the magnitude escalation between the two networks — all without delicate per-dataset tuning.

## Background

**The adversarial game.** A generator G maps a latent code z ∼ p(z) (conventionally N(0, I) or uniform) to an image; a discriminator (critic) D scores images. The original objective (Goodfellow et al. 2014) is the two-player minimax

  min_G max_D  E_{x∼data}[log D(x)] + E_{z}[log(1 − D(G(z)))],

whose inner optimum corresponds to minimizing the Jensen–Shannon divergence between the data and the generated distribution. The generator is the object of interest; the discriminator is an adaptive loss that is discarded after training. Networks are convolutional in the DCGAN style (Radford et al. 2015): strided / transposed convolutions, BatchNorm, (leaky) ReLU, weights initialized from N(0, 0.02²).

**Why the gradient dies at high resolution.** Arjovsky & Bottou (2017) showed that when the data manifold and the generated manifold have negligible overlap — generic for two low-dimensional manifolds embedded in a high-dimensional pixel space — a near-perfect discriminator exists, and the gradient it passes back to G either vanishes or points in essentially arbitrary directions. Odena et al. (2017) observed that increasing the output resolution makes generated images *easier* to distinguish from real ones, which drives the discriminator toward that near-perfect regime sooner. The two facts compound: the higher the resolution, the worse the gradient pathology.

**Stabilizing by changing the distance.** One line of work replaces the JS objective with a distance that stays informative even when the supports are disjoint. The Wasserstein (earth-mover) distance (Arjovsky et al. 2017) is continuous and differentiable almost everywhere in the generator's parameters under mild conditions, and is estimated by a critic D constrained to be 1-Lipschitz; the critic is *not* saturated by a perfect classifier and keeps producing a usable gradient. The improved variant (Gulrajani et al. 2017) enforces the Lipschitz constraint with a gradient penalty rather than weight clipping: it samples points x̂ on straight lines between real and generated images and penalizes the deviation of ‖∇_{x̂} D(x̂)‖₂ from 1,

  L_D = E_{z}[D(G(z))] − E_{x∼data}[D(x)] + λ · E_{x̂}[ (‖∇_{x̂} D(x̂)‖₂ − 1)² ],

with the generator minimizing −E_z[D(G(z))]. Other objectives in the same spirit are least-squares (Mao et al. 2016) and margin-based variants. These improvements are largely orthogonal to *how* the networks are built and grown.

**Variation and mode collapse.** Generators routinely cover only part of the data's variation. Salimans et al. (2016) proposed minibatch discrimination: let the discriminator look at statistics *across* the minibatch rather than at each image in isolation, so that a batch of near-identical generated images is detectable. Concretely, each feature vector f(x_i) is multiplied by a learned tensor T to give matrices M_i; for each sample one computes c(x_i) = Σ_j exp(−‖M_i − M_j‖_{L1}) summed over the batch, and these cross-sample statistics are concatenated to D's features. It helps but adds a learned tensor, extra hyperparameters, and a placement choice. Other routes include unrolling the discriminator (Metz et al. 2016) and a repelling regularizer (Zhao et al. 2016) that pushes the generator to orthogonalize feature vectors within a minibatch.

**Normalization and signal magnitudes.** Mode collapse often begins abruptly — within a dozen minibatches — when the discriminator overshoots, producing exaggerated gradients, after which signal magnitudes escalate in both networks. Most generators borrow BatchNorm (Ioffe & Szegedy 2015) — normalize each activation by minibatch mean/variance — sometimes with LayerNorm (Ba et al. 2016) or WeightNorm (Salimans & Kingma 2016) in the discriminator. These were introduced to combat internal covariate shift, a motivation that does not clearly apply here; the operative need in adversarial training is bounding signal magnitudes and the competition between the two nets. A related primitive is local response normalization (Krizhevsky et al. 2012), which divides each activation by an aggregate over neighboring feature channels at the same spatial location.

**Initialization for deep nets.** He et al. (2015) showed that initializing weights from N(0, gain²/fan_in) (gain = √2 for ReLU/leaky-ReLU) keeps activation variance roughly constant through depth, which is needed to train deep convolutional stacks. Adaptive optimizers — RMSProp (Tieleman & Hinton) and Adam (Kingma & Ba 2015) — normalize each parameter's update by a running estimate of its gradient's standard deviation, which makes the update step invariant to the overall scale of that parameter.

**Curriculum and multi-scale ideas.** Learning easy cases before hard ones (curriculum learning) and building images coarse-to-fine via image pyramids (Burt & Adelson 1987; Laplacian-pyramid generators, Denton et al. 2015) are long-standing tools; multi-scale generator/discriminator setups (Durugkar et al. 2016; Ghosh et al. 2017; Wang et al. 2017) attach discriminators at several resolutions.

## Baselines

- **GAN (Goodfellow et al. 2014).** The minimax JS-divergence game. Sharp samples but brittle training and low resolution; the gradient vanishes when D becomes confident. No mechanism for variation.

- **DCGAN (Radford et al. 2015).** The convolutional G/D backbone with BatchNorm and (leaky) ReLU that made low-resolution adversarial training reproducible. Gap: does not reach high-resolution, high-variation synthesis; stability degrades as resolution grows.

- **WGAN-GP (Gulrajani et al. 2017).** ResNet- or DCGAN-style critic trained with the Wasserstein distance and a gradient penalty enforcing 1-Lipschitzness. Supplies a usable gradient even when supports barely overlap — the strongest available stabilizer and a natural loss to build on. Gap: it fixes the *loss* but not the underlying difficulty that, at high resolution, learning the entire latent→image map at once is a very hard optimization problem with a near-perfect discriminator from the start; demonstrated mainly at modest resolution.

- **Minibatch discrimination (Salimans et al. 2016).** Adds cross-minibatch statistics to the discriminator to combat mode collapse via a learned projection tensor. Gap: introduces learned parameters and hyperparameters, requires choosing where to insert it, and in practice the gain in variation is limited.

The standard high-resolution attempt is to take a DCGAN/WGAN-GP setup, enlarge it to the target resolution, and train end-to-end. The gaps that leaves open: training is unstable and slow at high resolution; small minibatches (forced by memory) make it worse; variation is weak; and signal magnitudes escalate between the two networks.

## Evaluation settings

- **Datasets.** CelebA (face images) and LSUN categories (e.g. bedroom) for unsupervised image generation, typically at 128². CIFAR-10 (10 classes, 32×32) for inception-score comparison. The aspiration is output resolutions up to 1024², for which a sufficiently varied high-quality dataset must exist.

- **Metrics.** Inception Score (Salimans et al. 2016), IS = exp(E_x KL(p(y|x) ‖ p(y))) using a pretrained classifier, rewarding confident per-image class posteriors and a diverse class marginal. Multi-scale structural similarity, MS-SSIM (Odena et al. 2017; Wang et al. 2003), which measures variation *among generated outputs* but not similarity to the training set. Earth-mover / Wasserstein distance between local-patch distributions across scales is an available primitive for comparing appearance and variation at each spatial frequency.

- **Protocol.** Train the generator, then sample many images and compute the metrics against dataset statistics; optionally use an exponential moving average of the generator's weights when sampling for evaluation, which yields smoother results.

## Code framework

The primitives below already exist: a convolution and a dense layer, leaky ReLU, an Adam optimizer, the WGAN-GP discriminator/generator losses, nearest-neighbor upsampling and average-pool downsampling, and latent/real-image samplers. What does *not* exist yet is how to build and train the networks so that high-resolution synthesis is stable and varied: how the generator and discriminator should be structured and grown across resolutions, how a newly introduced piece should be brought in without disrupting what is already trained, what (if anything) replaces BatchNorm to keep magnitudes bounded, how weights should be initialized/scaled, and how to make the discriminator sensitive to a lack of variation. Those are the stubs below.

```python
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# --- existing primitives -------------------------------------------------

def leaky_relu(x, alpha=0.2):
    return torch.maximum(x * alpha, x)

def upscale2d(x, factor=2):           # nearest-neighbor (element replication)
    N, C, H, W = x.shape
    x = x.view(N, C, H, 1, W, 1).expand(N, C, H, factor, W, factor)
    return x.reshape(N, C, H * factor, W * factor)

def downscale2d(x, factor=2):         # average pooling
    return F.avg_pool2d(x, factor)

def wgan_gp_d_loss(D, real, fake, lam=10.0, drift=1e-3):
    # E[D(fake)] - E[D(real)] + lam*(||grad_xhat D||_2 - 1)^2 + drift*E[D(real)^2]
    ...

def wgan_g_loss(D, fake):
    return -D(fake).mean()

def sample_latents(batch, dim_z, device):
    z = torch.randn(batch, dim_z, device=device)
    return z

def sample_reals(batch):
    ...

# --- empty slots the method will fill ------------------------------------

class WeightedConv(nn.Module):
    """A conv/dense layer. TODO: how to initialize and scale its weights so every
       layer learns at the same effective rate under an adaptive optimizer."""
    def __init__(self, *a, **k):
        super().__init__()
        pass

def feature_norm(x):
    """TODO: a parameter-free normalization in the generator that bounds signal
       magnitudes (replacing BatchNorm), if one is needed at all."""
    pass

def variation_feature(x):
    """TODO: a parameter-free signal appended to the discriminator that lets it
       detect when a minibatch lacks variation."""
    pass

class Generator(nn.Module):
    """TODO: structure that can produce images at increasing resolutions, and a
       way to bring a higher-resolution stage online during training without
       disrupting the already-trained lower-resolution stages."""
    def __init__(self, dim_z, max_resolution, ch):
        super().__init__()
        pass
    def forward(self, z, lod):
        pass

class Discriminator(nn.Module):
    """TODO: mirror of the generator, grown in synchrony; reads images at the
       current resolution and outputs a scalar critic score."""
    def __init__(self, max_resolution, ch):
        super().__init__()
        pass
    def forward(self, x, lod):
        pass

def training_schedule(images_seen):
    """TODO: map training progress to the current resolution and to the blend
       weight of the stage currently being introduced."""
    pass

def train_step(G, D, G_opt, D_opt, lod, cfg):
    # TODO: process reals to the current resolution + blend; one D step (wgan_gp_d_loss),
    #       one G step (wgan_g_loss); maintain an EMA copy of G for sampling.
    pass
```
