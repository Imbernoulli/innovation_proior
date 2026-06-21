## Research question

A great many scientific measurement problems have the same shape. There is an unknown
signal `x_0` — a face image, a slice of brain anatomy, a permittivity map, a black-hole
photograph — and all we get to see is a degraded, noisy observation

```
y = f(x_0) + v,      v ~ N(0, sigma_v^2 I),
```

where the forward (measurement) operator `f` is known and may be linear (a mask, a
blur, a subsampled Fourier transform) or genuinely nonlinear (clipping, phase loss,
a scattering model). The map is many-to-one, so the problem is ill-posed: infinitely
many signals explain the same `y`, and recovering a plausible `x_0` requires a strong
prior on what real signals look like. In Bayesian terms we want to sample from the
posterior `p(x_0 | y) ∝ p(y | x_0) p(x_0)`, where the likelihood `p(y | x_0) =
N(f(x_0), sigma_v^2 I)` is given by the noise model and the prior `p(x_0)` is whatever
we know about the signal class.

By this time we have an extraordinarily good source of priors: pretrained denoising
diffusion models, trained once on a signal class (ImageNet faces, fastMRI brains) and
known to capture rich, multimodal structure. The question is how to build a *plug-and-play*
sampler — one that takes any such pretrained prior and any forward operator `f` and produces
posterior samples, **without retraining the prior for each new task**.

## Background

**Diffusion models and the score.** A diffusion model defines a forward noising
process that turns data into Gaussian noise,

```
x_t = alpha_t x_0 + sigma_t epsilon,    epsilon ~ N(0, I),   t in [0, T],
```

(for the variance-preserving choice, `sigma_t = sqrt(1 - e^{-∫_0^t beta(s) ds})` and
`alpha_t = sqrt(1 - sigma_t^2)`, so `x_T` is essentially `N(0, I)`). A network
`epsilon_theta(x_t; t)` is trained by denoising score matching to predict the added
noise, `min_theta E[|| epsilon - epsilon_theta(x_t; t) ||^2]`, and it is a scaled
estimate of the score of the *diffused* data, `epsilon_theta(x_t; t) ≈ -sigma_t
∇_{x_t} log p(x_t)`. Sampling runs the reverse SDE/ODE using this score (DDPM, DDIM,
and faster solvers). A standard and load-bearing fact is **Tweedie's / the MMSE
identity**: the posterior mean of the clean signal given a noisy `x_t` is computable in
closed form from the same network,

```
E[x_0 | x_t] = (1 / alpha_t) ( x_t - sigma_t epsilon_theta(x_t; t) ).
```

**Why the conditional score is the hard part.** To sample the *posterior*
`p(x_t | y)` along the diffusion trajectory, Bayes' rule splits the score into a prior
part and a likelihood part,

```
∇_{x_t} log p(x_t | y) = ∇_{x_t} log p(y | x_t) + ∇_{x_t} log p(x_t).
```

The prior score `∇_{x_t} log p(x_t)` is exactly what the pretrained network gives. The
likelihood score `∇_{x_t} log p(y | x_t)` requires computing

```
p(y | x_t) = ∫ p(y | x_0) p(x_0 | x_t) dx_0,
```

and the denoising posterior `p(x_0 | x_t)` is in general highly complex and multimodal,
so `p(y | x_t)` has no tractable closed form and cannot be estimated without
task-specific training. Every plug-and-play sampler is, at heart, a different way of
coping with this one intractable term.

**Score-matching is a divergence, with a known weighting.** A second body of theory
will matter. Work on maximum-likelihood training of score-based models (Song et al.
2021) established that a *KL divergence between two diffusion-coupled distributions
equals a time-integrated, weighted score-matching loss* along the trajectory: for
distributions `q` and `p` diffused by the same SDE,

```
KL(q(x_0) || p(x_0)) = ∫_0^T (beta(t)/2) E_{q(x_t)} [ || ∇_{x_t} log q(x_t)
                                                     - ∇_{x_t} log p(x_t) ||^2 ] dt,
```

(the latent-space version is the cross-entropy theorem of Vahdat et al. 2021). The same
line of work supplies an integration-by-parts identity — `d KL(q(x_t)||p(x_t)) / dt =
-(beta(t)/2) E_q[ ||∇log q - ∇log p||^2 ]` — that lets one *re-weight* the
score-matching integral over time by an arbitrary positive function `omega(t)` while
controlling the boundary terms. These are pre-existing tools about diffusion processes,
not about inverse problems.

**An observed pathology of the residual.** It is documented for VP diffusion that the
noise residual `||epsilon_theta(x_t; t) - epsilon||^2`, and the corresponding score
mismatch, behaves very unevenly across `t`: it stays moderate at large noise but grows
sharply as `t -> 0` (the signal-to-noise ratio climbs much faster than the residual
shrinks). Equivalently, the per-timestep contributions of a diffusion objective are
badly scale-mismatched across the trajectory unless reweighted — different timesteps are
known to govern different image content, from large-scale structure at high noise to
fine detail at low noise. Any procedure that sums denoiser feedback over many noise
levels has to reckon with this imbalance.

**Variational inference.** The classical alternative to sampling-by-simulation is
variational inference: posit a tractable family `q` and fit it to an intractable target
by minimizing a KL divergence / maximizing an evidence lower bound (Blei et al. 2017;
the VAE objective of Kingma & Welling 2013; Rezende et al. 2014). For a Gaussian `q`
the relevant expectations are often closed-form or cheaply reparameterizable. This is a
mode-seeking framework: `KL(q || p)` concentrates `q` on a dominant mode of the target.

## Baselines

**DPS — Diffusion Posterior Sampling (Chung et al. 2022).** Attacks the intractable
likelihood score head-on by approximating `p(y | x_t)` with `p(y | x̂_0)`, where
`x̂_0 = E[x_0 | x_t]` is the Tweedie posterior mean. Under Gaussian noise this gives an
analytic guidance term, and the conditional score becomes

```
∇_{x_t} log p(x_t | y) ≈ s_theta(x_t, t) - (1/sigma_v^2) ∇_{x_t} || y - f(x̂_0(x_t)) ||^2,
```

plugged into a standard reverse-diffusion step. It is general — it handles nonlinear
`f` and noisy `y`. Because `x̂_0(x_t)` is a function of `x_t` through the network, the
guidance gradient `∇_{x_t}|| y - f(x̂_0(x_t)) ||^2` requires **backpropagating through
the diffusion denoiser at every step** (a score-Jacobian / vector-Jacobian product).

**ΠGDM — Pseudoinverse-Guided Diffusion (Song et al. 2023).** Sharpens the guidance by
modeling `p(x_0 | x_t)` as a Gaussian around `x̂_0` and folding the measurement operator
in through its pseudoinverse, which improves the guidance approximation relative to
DPS. It too provides a likelihood-score surrogate that is added to the prior score, and
differentiates through the model.

**RED — Regularization by Denoising (Romano, Elad & Milanfar 2016).** A classical, very
different idea: use *any* image denoiser `f_den` as an explicit regularizer. Define the
energy

```
E(x) = ℓ(y, x) + (lambda/2) x^T ( x - f_den(x) ),
```

an image-adaptive Laplacian penalty whose value is small when `x` is a fixed point of
the denoiser (`x ≈ f_den(x)`) or when the residual `x - f_den(x)` is orthogonal to `x`
(behaves like white noise). Its celebrated property: under a local-homogeneity
condition on the denoiser (`∇f_den(x) x = f_den(x)`), the gradient of the regularizer
collapses to

```
∇_x [(1/2) x^T (x - f_den(x))] = x - f_den(x),
```

so minimizing `E` by gradient descent needs **only the denoiser applied once per step —
no differentiation of the denoiser**. This is the structural advantage worth noticing:
a denoiser-based regularizer whose gradient is just a residual. (A closely related
observation: Reehorst & Schniter 2018 connect RED's residual to score matching for a
single denoiser.)

**PnP-ADMM / Plug-and-Play priors (Venkatakrishnan et al. 2013).** Inserts a denoiser
as the proximal operator inside ADMM, using the denoiser as an implicit prior at each
ADMM subproblem step.

## Evaluation settings

The natural yardsticks, all pre-existing:

- **Linear image restoration** on a 1k validation subset of `256x256` ImageNet (and
  FFHQ faces): box / random-mask **inpainting** (Palette masks), `4x`
  **super-resolution** (bicubic degradation). The prior is a publicly available
  unconditional guided-diffusion checkpoint.
- **Nonlinear tasks**: high-dynamic-range clipping, phase retrieval, nonlinear
  deblurring — to stress operators that linear-only methods cannot handle.
- **Noisy observations**: additive Gaussian measurement noise (e.g. `sigma_v` around
  0.05-0.1) on the masked/observed pixels.
- **Scientific inverse problems** with the same harness: compressed-sensing MRI
  (fastMRI brain, mridata knee, with undersampling masks), and operators such as
  inverse scattering / interferometric (sparse-Fourier) imaging.
- **Metrics**: PSNR and SSIM for fidelity; LPIPS and KID for perceptual quality; for
  some scientific tasks domain metrics (e.g. closure-phase chi-squared). Also wall-clock
  per step and peak GPU memory, since the differentiate-through-the-network methods are
  expensive. Protocol: a fixed number of diffusion/optimization steps (commonly 100 or
  1000), each method tuned to its best hyperparameters.

## Code framework

The substrate is everything that exists *before* a new sampler: a frozen pretrained
denoiser, the forward operator with its data-fitting gradient, a noise schedule, and an
off-the-shelf optimizer. The open design question is the body of `inference`: how to
combine these pieces into an estimate of `x_0` from `y`.

```python
class PosteriorSampler:
    """Plug-and-play reconstruction from y = f(x_0) + noise using frozen components."""

    def __init__(self, net, forward_op, scheduler, optimizer_cls, **kwargs):
        self.net = net.eval().requires_grad_(False)
        self.forward_op = forward_op
        self.scheduler = scheduler
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs

    def inference(self, observation, num_samples=1):
        # TODO: design the reconstruction rule that combines these primitives.
        raise NotImplementedError
```

The single empty slot is the reconstruction procedure itself: every primitive it might
use already exists, but the rule that combines them is exactly what is to be designed.
