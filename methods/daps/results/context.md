# Context: posterior sampling for diffusion inverse problems

## Research question

A pretrained diffusion model encodes a prior `p(x_0)` over a signal class, available as a denoiser
`D_theta(x_t, sigma)` that returns the Tweedie posterior mean `E[x_0 | x_t]` at noise level `sigma`.
Given a known forward operator `A` and a noisy observation `y = A(x_0) + n`, `n ~ N(0, sigma_y^2 I)`,
we want posterior samples `x_0 ~ p(x_0 | y) ∝ p(y | x_0) p(x_0)`, training-free, for general — including
strongly nonlinear — `A` (phase retrieval, nonlinear deblurring, high dynamic range). The central
quantity is the time-level likelihood score `∇_{x_t} log p_t(y | x_t)`, which is intractable because `y`
depends on the noised iterate `x_t` only through the unknown clean `x_0`. The question is how to draw
posterior samples using only the pretrained denoiser and the known forward model.

## Background

**Diffusion models and the score.** A forward process `x_t = x_0 + sigma_t epsilon` (the
variance-exploding / EDM convention; the variance-preserving form rescales) carries data to noise; a
network trained by denoising score matching gives the score `∇_{x_t} log p_t(x_t)`, and equivalently the
Tweedie/MMSE estimate `D_theta(x_t, sigma) = E[x_0 | x_t] = x_t + sigma^2 ∇_{x_t} log p_t(x_t)`. Sampling
runs the reverse SDE or the probability-flow ODE; EDM (Karras et al. 2022) fixes the standard time
discretization and the rho-schedule of noise levels from `sigma_max` to `sigma_min`.

**The conditional score.** To condition on `y` along the trajectory, Bayes splits the score:
`∇_{x_t} log p_t(x_t | y) = ∇_{x_t} log p_t(x_t) + ∇_{x_t} log p_t(y | x_t)`. The prior score is free; the
likelihood score needs `p_t(y | x_t) = ∫ p(y | x_0) p(x_0 | x_t) dx_0`, and the denoising posterior
`p(x_0 | x_t)` is multimodal, so the likelihood at the noised iterate has no closed form.

**Two variables.** The two factors of posterior sampling live on different variables: the diffusion prior
defines a score on the *noisy* latent `x_t`, while the likelihood `p(y | x_0)` is defined on the *clean*
`x_0`. The guidance scale that mediates between prior dynamics and likelihood correction is typically
retuned per operator.

## Baselines

These are the training-free diffusion solvers available for inverse problems.

**DPS — Diffusion Posterior Sampling (Chung et al. 2023; arXiv:2209.14687).** Approximates
`p_t(y | x_t) ≈ p(y | x_0_hat)` with the Tweedie mean `x_0_hat`, then adds the guidance
`∇_{x_t} log p(y | x_0_hat)` — a data-fidelity loss backpropagated through the denoiser — to each reverse
step. General and nonlinear-capable; costs a denoiser Jacobian per step.

**ΠGDM (Song et al. 2023).** Sharpens the guidance by modeling `p(x_0 | x_t)` as a Gaussian and inverting
`A` through its pseudoinverse, applied to linear and semi-linear operators; it differentiates through the
network.

**LGD — Loss-Guided Diffusion (Song et al. 2023).** Replaces DPS's delta with a Gaussian surrogate
`N(x_0_hat, r_t^2 I)`, `r_t = sigma_t/sqrt(1+sigma_t^2)`, and Monte-Carlo-estimates the guidance over `n`
samples branching off one denoiser call. It is a per-step guidance term added to the reverse update and
uses a denoiser Jacobian.

**RED-diff (Mardani et al. 2024; arXiv:2305.04391).** Steps off the trajectory entirely: fit a Gaussian
`q(x_0 | y)` to the posterior by minimizing `KL(q || p(x_0|y))`, whose prior-KL becomes a score-matching
loss whose gradient is the denoising residual `epsilon_theta − epsilon` — no denoiser Jacobian, solved by
Adam on the clean `mu`. It optimizes a single Gaussian via mode-seeking KL, and the noise residual is
weighted by `1/SNR`.

## Evaluation settings

The standard yardsticks for an image-domain diffusion inverse-problem solver:

- **Datasets / priors.** FFHQ `256x256` and ImageNet `256x256`, held-out validation images, shared
  pretrained unconditional score networks, images normalized to `[0,1]`.
- **Linear operators.** Super-resolution (`4x`), Gaussian and motion deblurring, box and random
  inpainting.
- **Nonlinear operators.** Fourier phase retrieval (`A(x) = |F x|`), nonlinear deblurring through a
  distilled forward model, high-dynamic-range reconstruction.
- **Noise.** Additive white Gaussian, `sigma_y = 0.05` on the `[0,1]` scale.
- **Metrics.** LPIPS and FID (perceptual), PSNR / SSIM (distortion); lower LPIPS/FID and higher PSNR/SSIM
  are better. Sampling cost in number of function evaluations (NFE).

## Code framework

The sampler plugs into the EDM-style diffusion scaffold: a `Scheduler` precomputing the annealing noise
levels and the per-step reverse-update coefficients, a pretrained denoiser `net` returning the Tweedie
mean `E[x_0 | x_t]`, an unconditional probability-flow ODE sampler, and a `forward_op` exposing `A`, its
squared-residual gradient, and the loss.

```python
import torch
import numpy as np
from .base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler


class Solver(Algo):
    """Sample x_0 given y, using a pretrained diffusion prior `net` (returns the
    Tweedie mean E[x_0 | x_t]) and a known forward operator `forward_op`. An
    unconditional ODE sub-sampler exists."""

    def __init__(self, net, forward_op, annealing_scheduler_config, diffusion_scheduler_config,
                 lgvd_lr, lgvd_steps, sigma_y):
        super().__init__(net, forward_op)
        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.lr = lgvd_lr
        self.steps = lgvd_steps
        self.sigma_y = sigma_y

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        # start from pure noise at the maximum annealing level
        x_t = torch.randn(num_samples, self.net.img_channels,
                          self.net.img_resolution, self.net.img_resolution,
                          device=device) * self.annealing_scheduler.sigma_max

        for i in range(self.annealing_scheduler.num_steps):
            sigma = self.annealing_scheduler.sigma_steps[i]
            sigma_next = self.annealing_scheduler.sigma_steps[i + 1]
            # TODO: combine the pretrained prior with the observation likelihood
            #       at each annealing level.
            pass

        return x_t
```

The harness supplies the annealing schedule, the unconditional ODE sub-sampler that turns `x_t` into a
clean estimate, the pretrained denoiser, and the forward operator's squared-residual gradient/loss.
