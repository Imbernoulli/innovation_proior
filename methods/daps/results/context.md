# Context: decoupled annealing posterior sampling for diffusion inverse problems

## Research question

A pretrained diffusion model encodes a prior `p(x_0)` over a signal class, available as a denoiser
`D_theta(x_t, sigma)` that returns the Tweedie posterior mean `E[x_0 | x_t]` at noise level `sigma`.
Given a known forward operator `A` and a noisy observation `y = A(x_0) + n`, `n ~ N(0, sigma_y^2 I)`,
we want posterior samples `x_0 ~ p(x_0 | y) ∝ p(y | x_0) p(x_0)`, training-free, for general — including
strongly nonlinear — `A`. The recurring obstacle is the time-level likelihood score
`∇_{x_t} log p_t(y | x_t)`, intractable because `y` depends on the noised iterate `x_t` only through the
unknown clean `x_0`.

The precise goal is a sampler that (1) uses only the pretrained denoiser and the known forward model;
(2) is stable and accurate on *nonlinear* problems (phase retrieval, nonlinear deblurring, high dynamic
range), where the existing trajectory-guidance solvers are most fragile; (3) needs few interpretable
knobs. The methods below each cope with the intractable term differently; the ones that handle general
nonlinear `A` do so by approximating a guidance term *inside* the reverse trajectory, and that coupling
is exactly where they become brittle. Closing that gap is the problem.

## Background

**Diffusion models and the score.** A forward process `x_t = x_0 + sigma_t epsilon` (the
variance-exploding / EDM convention; the variance-preserving form rescales) carries data to noise; a
network trained by denoising score matching gives the score `∇_{x_t} log p_t(x_t)`, and equivalently the
Tweedie/MMSE estimate `D_theta(x_t, sigma) = E[x_0 | x_t] = x_t + sigma^2 ∇_{x_t} log p_t(x_t)`. Sampling
runs the reverse SDE or the probability-flow ODE; EDM (Karras et al. 2022) fixes the standard time
discretization and the rho-schedule of noise levels from `sigma_max` to `sigma_min`.

**Why the conditional score is the hard part.** To condition on `y` along the trajectory, Bayes splits
the score: `∇_{x_t} log p_t(x_t | y) = ∇_{x_t} log p_t(x_t) + ∇_{x_t} log p_t(y | x_t)`. The prior score
is free; the likelihood score needs `p_t(y | x_t) = ∫ p(y | x_0) p(x_0 | x_t) dx_0`, and the denoising
posterior `p(x_0 | x_t)` is multimodal, so the likelihood at the noised iterate has no closed form.

**The decoupling observation.** The two factors of posterior sampling live on different variables: the
diffusion prior defines a score on the *noisy* latent `x_t`, while the likelihood `p(y | x_0)` is defined
on the *clean* `x_0`. Methods that add a guidance term to a single reverse step force both onto `x_t` at
once, coupling the prior update and the likelihood correction; the consistency the likelihood demands and
the dynamics the prior demands then fight, and the guidance scale must be retuned per operator.

## Baselines

These are the training-free diffusion solvers a decoupled sampler has to improve on.

**DPS — Diffusion Posterior Sampling (Chung et al. 2023; arXiv:2209.14687).** Approximates
`p_t(y | x_t) ≈ p(y | x_0_hat)` with the Tweedie mean `x_0_hat`, then adds the guidance
`∇_{x_t} log p(y | x_0_hat)` — a data-fidelity loss backpropagated through the denoiser — to each reverse
step. General and nonlinear-capable. Gap: collapsing the denoising posterior to its mean is a Jensen
point estimate, least reliable at high noise where the posterior is broad; and it costs a denoiser
Jacobian per step, making it sensitive to the guidance scale and step schedule.

**ΠGDM (Song et al. 2023).** Sharpens the guidance by modeling `p(x_0 | x_t)` as a Gaussian and inverting
`A` through its pseudoinverse. Gap: the pseudoinverse specializes it to linear (and semi-linear)
operators; general nonlinear `A` is out, and it still differentiates through the network.

**LGD — Loss-Guided Diffusion (Song et al. 2023).** Replaces DPS's delta with a Gaussian surrogate
`N(x_0_hat, r_t^2 I)`, `r_t = sigma_t/sqrt(1+sigma_t^2)`, and Monte-Carlo-estimates the guidance over `n`
samples branching off one denoiser call. Reduces the point-estimate bias. Gap: still a per-step guidance
term bending the *coupled* reverse update, still a denoiser Jacobian; the sampled spread can perturb
operators DPS already handled well.

**RED-diff (Mardani et al. 2024; arXiv:2305.04391).** Steps off the trajectory entirely: fit a Gaussian
`q(x_0 | y)` to the posterior by minimizing `KL(q || p(x_0|y))`, whose prior-KL becomes a score-matching
loss whose gradient is the denoising residual `epsilon_theta − epsilon` — no denoiser Jacobian, solved by
Adam on the clean `mu`. Gap: it optimizes a single Gaussian (effectively a point with `sigma → 0`) by
mode-seeking KL, so it can under-explore and blur on multimodal posteriors, and the noise residual needs
a `1/SNR` weighting to be stable.

**The diagnostic that motivates the method.** Across these, the methods that keep nonlinearity generality
(DPS, LGD) couple the prior step and the likelihood correction on `x_t` and approximate a trajectory
guidance term; the method that decouples (RED-diff) commits to a single mode and an optimization rather
than a *sampler*. A method that both (i) decouples the prior update from the likelihood correction — doing
the likelihood correction on a *clean* estimate, then re-noising — and (ii) remains a proper sampler with
stochastic exploration of the clean variable, would keep nonlinearity generality without the coupled-step
fragility. That is the gap.

## Evaluation settings

The standard yardsticks for an image-domain diffusion inverse-problem solver:

- **Datasets / priors.** FFHQ `256x256` and ImageNet `256x256`, held-out validation images, shared
  pretrained unconditional score networks, images normalized to `[0,1]`.
- **Linear operators.** Super-resolution (`4x`), Gaussian and motion deblurring, box and random
  inpainting.
- **Nonlinear operators.** Fourier phase retrieval (`A(x) = |F x|`), nonlinear deblurring through a
  distilled forward model, high-dynamic-range reconstruction — the regime that breaks trajectory-guidance
  solvers.
- **Noise.** Additive white Gaussian, `sigma_y = 0.05` on the `[0,1]` scale.
- **Metrics.** LPIPS and FID (perceptual), PSNR / SSIM (distortion); lower LPIPS/FID and higher PSNR/SSIM
  are better. Sampling cost in number of function evaluations (NFE).

## Code framework

The sampler plugs into the EDM-style diffusion scaffold: a `Scheduler` precomputing the annealing noise
levels and the per-step reverse-update coefficients, a pretrained denoiser `net` returning the Tweedie
mean `E[x_0 | x_t]`, an unconditional probability-flow ODE sampler, and a `forward_op` exposing `A`, its
squared-residual gradient, and the loss. The unconditional sub-sampler is given; what is *not* settled is
how to interleave the prior update and the likelihood correction so the two stop fighting.

```python
import torch
import numpy as np
from .base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler


class Solver(Algo):
    """Sample x_0 given y, using a pretrained diffusion prior `net` (returns the
    Tweedie mean E[x_0 | x_t]) and a known forward operator `forward_op`. An
    unconditional ODE sub-sampler exists; the prior/likelihood interleaving does not."""

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

            # 1) PRIOR UPDATE: unconditional x0hat = E[x0 | x_t] by solving the
            #    probability-flow ODE from x_t (a diffusion sub-sampler), NOT one Tweedie call.
            # 2) LIKELIHOOD CORRECTION on the CLEAN variable: sample p(x0 | x_t, y) by Langevin.
            # 3) RE-NOISE: x_{t-1} ~ N(x0_sample, sigma_next^2 I) for the next level.
            # TODO: interleave (1)-(3) so the prior update and likelihood correction decouple.
            pass

        return x_t
```

The harness supplies the annealing schedule, the unconditional ODE sub-sampler that turns `x_t` into a
clean estimate, and the forward operator's squared-residual gradient/loss; the `# TODO` is the one place
the decoupled prior-update / likelihood-correction / re-noise cycle lives.
