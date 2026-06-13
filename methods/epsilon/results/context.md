# Context: generative modeling by learned denoising (circa 2019-2020)

## Research question

We want a generative model of natural images that is **simple to define, stable to train, and
produces sharp, high-fidelity samples** — and ideally one that is a proper likelihood model
(so it can be evaluated and compared), not just a sample generator. The dominant high-quality
samplers of the time are adversarial (GANs): they make beautiful images but train unstably and
give no tractable likelihood for model comparison. The dominant likelihood models
(autoregressive nets, normalizing flows) are stable and give exact likelihoods but need
specialized, constrained architectures and, for images, sample slowly one coordinate at a time.

There is a third idea that is conceptually clean — define a fixed forward process that
gradually destroys the data into noise, then *learn the reverse* — but to date it has produced
only blurry, low-quality images, never anything competitive. The concrete problem is: given a
**fixed** Gaussian corruption process, choose the parameterization of the learned reverse step
and the per-step training loss so that a finite-time reverse Markov chain, trained by a
tractable objective, actually generates high-quality images. Two design choices are wide open
and entangled: *what the reverse network should output*, and *how each timestep's loss should
be weighted*. The pain point is that the obvious answers (have the network predict the reverse
mean directly; train on the exact variational bound) have already been tried and give weak
samples — so the open question is whether a different but equivalent reparameterization, or a
different loss weighting, unlocks the sample quality that the architecture is clearly capable
of.

## Background

**Latent-variable models and the variational bound.** A latent-variable model defines
`p_theta(x_0) = ∫ p_theta(x_0, z) dz` and is trained by maximizing the evidence lower bound
(ELBO). With an approximate posterior `q(z|x_0)`,
`log p_theta(x_0) ≥ E_q[ log p_theta(x_0,z) − log q(z|x_0) ]`. Kingma & Welling (2013)
made this practical for continuous latents with the **reparameterization trick**: write a
Gaussian sample as `z = mu + sigma·eps`, `eps ~ N(0,I)`, so that gradients flow through the
sampling. When both `q` and `p` are Gaussian, the per-term KL divergence has a closed form, so
the bound can be optimized by low-variance stochastic gradients rather than high-variance Monte
Carlo. These two facts — reparameterized Gaussian sampling and closed-form Gaussian KL — are
the workhorses of everything below.

**Diffusion as a fixed forward process with a learnable reverse.** Sohl-Dickstein et al.
(2015) proposed defining the model as the endpoint of a Markov chain. A **forward (diffusion)
process** gradually adds Gaussian noise to the data over `T` steps according to a variance
schedule `beta_1,...,beta_T`,

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) x_{t-1}, beta_t I),
```

slowly converting any data distribution into an isotropic Gaussian. The **reverse process**
`p_theta(x_{t-1}|x_t)` is learned to undo one step. Their key structural observation, borrowed
from nonequilibrium thermodynamics and the Feller/Kolmogorov theory of diffusions: *when each
forward step adds only a small amount of noise (`beta_t` small), the reverse conditional has
the same functional form as the forward one* — a Gaussian. So it suffices to let the reverse
step be a Gaussian whose mean and variance are functions of `(x_t, t)`. Training maximizes a
variational bound on `−log p_theta(x_0)`, and the model can be made arbitrarily expressive by
taking many small steps. A second useful property: each forward step *scales the signal down*
by `sqrt(1-beta_t)` as it injects variance `beta_t`, so the total variance stays bounded and
the chain converges to a standard normal. Sohl-Dickstein et al. also derived **entropy bounds**
that pin down two natural choices for the reverse variance — equal to the forward step variance,
or equal to the variance of the true reverse posterior — as the upper and lower extremes for
data with unit coordinate variance. The method was elegant and tractable but had never been
shown to make high-quality images.

**The score / Langevin line.** A parallel approach models the **score** of the data density,
`∇_x log p(x)` — the vector field pointing toward higher density — instead of the density
itself. Given the score, **Langevin dynamics** generates samples by the recursion
`x_{t} = x_{t-1} + (eps/2)·∇_x log p(x_{t-1}) + sqrt(eps)·z_t`, `z_t ~ N(0,I)`, which converges
to `p(x)` as the step size shrinks and the number of steps grows. The attraction is that
sampling needs only the score, never the normalizing constant. **Score matching** learns the
score directly. Hyvärinen's original objective involves the trace of the Jacobian of the score
network, which does not scale to high dimensions; **denoising score matching** (Vincent, 2011)
removes that cost. Vincent's identity is the load-bearing fact: if a clean point `x` is
corrupted by `q_sigma(x̃|x) = N(x̃; x, sigma^2 I)`, then the score of the *corruption kernel*
is available in closed form,

```
∇_{x̃} log q_sigma(x̃ | x) = −(x̃ − x) / sigma^2,
```

and matching the network's output to this target trains it to estimate the score of the
*noised* data density `q_sigma(x̃)`. Concretely, the denoising score-matching loss is
`½ E[ ‖ s_theta(x̃) − ∇_{x̃} log q_sigma(x̃|x) ‖^2 ]`, i.e. predicting `−(x̃−x)/sigma^2`, which
is the (negated, rescaled) corruption noise. So *denoising a Gaussian corruption is the same as
estimating a score.*

**The empirical phenomena that motivate noising at many scales.** Song & Ermon (2019, NCSN)
documented two concrete failure modes of naive score-based generation, both observed
empirically and both explainable from first principles. First, the **manifold hypothesis**:
real image data concentrate near a low-dimensional manifold, so `∇_x log p_data(x)` is
ill-defined off the manifold and the score-matching estimator is inconsistent; they show the
sliced-score-matching training loss on clean CIFAR-10 fails to converge (it fluctuates), but
adding even tiny Gaussian noise `N(0, 0.0001)` — imperceptible to the eye — makes the loss
converge cleanly, because the perturbed density has full support. Second, **low-density
regions**: where there are few data samples the score is estimated poorly, and Langevin
dynamics initialized in such regions, or needing to cross them between modes, mixes far too
slowly and gets the relative weights of separated modes wrong (demonstrated on a two-Gaussian
mixture, where Langevin with the *exact* score still recovers the wrong mode weights). The fix
NCSN adopts is to perturb the data at a **geometric ladder of noise levels**
`sigma_1 > ... > sigma_L`, train one **noise-conditional** score network `s_theta(x, sigma)` to
estimate the score at every level via denoising score matching, and sample by **annealed
Langevin dynamics** — start at the largest noise (smooth, easy to traverse) and walk the level
down to the smallest (close to the data). They also note empirically that at optimum
`‖s_theta(x,sigma)‖ ∝ 1/sigma`, which motivates weighting level `sigma`'s loss by
`lambda(sigma) = sigma^2`; under that weight the per-level loss becomes
`½ E[ ‖ sigma·s_theta(x̃,sigma) + (x̃−x)/sigma ‖^2 ]`, whose magnitude is the same across
levels because `(x̃−x)/sigma ~ N(0,I)` and `‖sigma·s_theta‖ ∝ 1`.

These two lines — a fixed Gaussian forward process trained by a variational bound, and a
multi-noise-level denoiser trained by score matching and sampled by annealed Langevin — are
developed completely separately, in different languages (variational inference vs. score
estimation), and it is not at all obvious that they are related.

## Baselines

These are the prior methods a new image generator would be measured against and would react to.

**GANs (Goodfellow et al. 2014; ProgressiveGAN, BigGAN, StyleGAN2).** A generator is trained
adversarially against a discriminator. Core idea: minimize a divergence between model and data
implicitly through the discriminator's signal. They produce the sharpest images of the era.
**Gap:** training is unstable (mode collapse, sensitivity to hyperparameters and
architecture), and there is no tractable likelihood, so models cannot be compared on a
principled probabilistic metric and there is no inductive guarantee that the generator covers
the data distribution.

**Autoregressive models (PixelCNN++, Gated PixelCNN, Sparse Transformer).** Factor the image
likelihood as a product of per-pixel conditionals and model each with a deep net. Core idea:
exact likelihood via the chain rule, trained by maximum likelihood. **Gap:** sampling is
inherently sequential — one coordinate at a time, `D` network evaluations for `D` pixels — and
the fixed raster ordering bakes in an inductive bias; high-resolution sampling is slow.

**Normalizing flows (NICE, RealNVP, Glow).** Build an invertible map from data to a simple
latent with a tractable Jacobian, giving exact likelihood. **Gap:** invertibility plus a
cheap-Jacobian constraint restricts the architecture, and sample quality has lagged GANs.

**The original diffusion model (Sohl-Dickstein et al. 2015).** The fixed-forward /
learned-reverse construction above, trained on the variational bound with the reverse step's
mean and variance both learned by a network. Core idea: model the data as the endpoint of a
learned reverse diffusion; tractable per-step Gaussians; exact-ish likelihood from the bound.
**Gap:** in practice it produced only low-quality, blurry samples — it had never been shown
that this family is capable of competitive image synthesis, and it left open exactly *how* to
parameterize the reverse Gaussian's mean (predict the mean directly? predict the clean image?)
and *how* to weight the per-timestep loss terms, with the naive choices giving weak results.

**NCSN (Song & Ermon 2019).** Multi-noise-level denoising score matching with annealed
Langevin sampling, described above; reached an unconditional CIFAR-10 Inception score of 8.87.
Core idea: estimate the score at many noise scales with one conditional network, sample by
annealing. **Gap:** the sampler's coefficients (step sizes, noise scales, number of steps) are
set by hand *after* training rather than derived from the training process, so training does
not directly optimize the quality of what the sampler produces; the noise scaling does not
shrink the signal (variance grows with noise), and the model is not a likelihood model and
matches the data only approximately. It is a denoiser plus a hand-tuned MCMC, not a single
trained generative chain with a bound it provably optimizes.

## Evaluation settings

The natural yardsticks already in use for unconditional image generation:

- **Datasets.** Unconditional CIFAR-10 (32×32, 50k train images) is the standard small-image
  generative benchmark; higher-resolution work uses LSUN categories (Church, Bedroom, Cat) and
  CelebA-HQ at 256×256. Image pixels in `{0,...,255}` are scaled linearly to `[-1, 1]` so the
  network sees consistently scaled inputs matching a standard-normal prior.
- **Metrics.** Inception Score (IS) and Fréchet Inception Distance (FID), both computed on
  50,000 generated samples against the dataset statistics (FID against the training set is
  standard; lower is better). For likelihood models, negative log likelihood in bits/dim.
- **Protocol.** Fixed network backbone and training budget across the parameterizations being
  compared; the forward noise schedule, number of diffusion steps, optimizer, and metric
  computation are held fixed so that only the reverse-process parameterization and the loss
  vary. Standard sampling uses the full `T`-step reverse chain.

## Code framework

The training substrate is a standard latent-variable / denoising training loop. The forward
Gaussian diffusion is **fixed** (no learnable parameters), so its schedule tensors are
precomputed once: with `alpha_t = 1 - beta_t` and `alpha_bar_t = prod_{s<=t} alpha_s`, the
closed-form one-shot corruption `q(x_t|x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)`
lets us jump to any timestep directly. The network is a time-conditioned image-to-image model
(a U-Net) that already exists; the optimizer (Adam), the MSE training loss, the data pipeline,
and the EMA of weights all already exist. What is **not** settled — and is exactly what must be
designed — is the bridge between the network's raw output and the diffusion: *what target the
network is trained to produce at each noisy sample*, and *how that output is turned back into a
prediction of the clean image for the reverse step*. Those two are a matched pair and must
invert each other. They are the single empty slot below.

```python
import torch


def get_schedule(betas):
    """Precompute the fixed forward-diffusion tensors (no learnable parameters)."""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alpha": alphas_cumprod.sqrt(),                 # sqrt(alpha_bar_t)
        "sqrt_one_minus_alpha": (1.0 - alphas_cumprod).sqrt(),  # sqrt(1 - alpha_bar_t)
    }


def q_sample(x_0, noise, t, schedule):
    """Forward process: x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) noise."""
    sa = schedule["sqrt_alpha"][t].view(-1, 1, 1, 1)
    soma = schedule["sqrt_one_minus_alpha"][t].view(-1, 1, 1, 1)
    return sa * x_0 + soma * noise


def compute_training_target(x_0, noise, t, schedule):
    """The target the network is trained to predict at the noisy sample x_t.

    TODO: choose the reverse-process parameterization's training target.
    """
    pass  # TODO


def predict_x0(model_output, x_t, t, schedule):
    """Invert the parameterization: recover the predicted clean image from the
    network output, for use in the reverse/sampling step. Must be the exact
    inverse of compute_training_target.

    TODO: choose the matching inversion.
    """
    pass  # TODO


def train_step(model, x_0, schedule, T):
    """Existing denoising training loop."""
    B = x_0.shape[0]
    t = torch.randint(0, T, (B,), device=x_0.device)         # random timestep per example
    noise = torch.randn_like(x_0)                            # eps ~ N(0, I)
    x_t = q_sample(x_0, noise, t, schedule)                  # corrupt to level t
    target = compute_training_target(x_0, noise, t, schedule)
    pred = model(x_t, t)                                     # time-conditioned U-Net
    loss = ((pred - target) ** 2).mean()                    # existing MSE loss
    return loss
```

The reverse/sampling step consumes `predict_x0` to take one denoising move from `x_t` toward
`x_{t-1}`; training and sampling share the same parameterization through these two stubs.
