## Research question

A pretrained diffusion model has learned a rich prior `p(x)` over a signal class — encoded as a
denoiser `net(x_t, sigma)` that returns the Tweedie posterior mean `E[x_0 | x_t]` at noise level
`sigma`. Separately there is an inverse problem: a known forward operator `A` and an observation
`y = A(x) + noise`, and the goal is to reconstruct `x` from `y` by conditioning that diffusion prior on
the measurement. The single thing being designed is the **measurement-conditioning rule**: how each
piece the harness exposes — the unconditional denoiser, the noise schedule, the forward operator and
its gradient — is combined into a sampler that turns `p(x)` into a draw from the posterior
`p(x | y) ∝ p(y | x) p(x)`. Everything else (the denoiser weights, the forward-operator definitions,
the evaluation problems) is fixed; the algorithm only chooses how to combine these pieces.

The conditioning rule has to hold up across three very different scientific inverse problems with very
different forward operators — a differentiable electromagnetic scattering model, a sparse radio
interferometer, a pixel mask — so it must work for general (including nonlinear) `A`, must not blow up
when the measurement is noisy, and must need little per-problem tuning.

## Prior art before the first rung (diffusion inverse-problem lineage)

The first rung reacts to a line of training-free diffusion solvers, each of which copes differently with
the one intractable object — the *time-level likelihood score* `∇_{x_t} log p_t(y | x_t)`, which has no
closed form because `y` depends on the noised iterate `x_t` only through the unknown clean `x_0`.

- **Reverse-SDE / score-based generative modeling (Song et al. 2021; Anderson 1982).** Generation is the
  reverse diffusion driven by the score `∇_{x_t} log p_t(x_t)`, which the denoiser supplies. To condition
  on `y`, Bayes splits the score into the prior score (free) plus the likelihood score (the wall). Gap:
  gives the decomposition but not the intractable likelihood term.
- **Projection-onto-measurement-subspace solvers (score-SDE; ILVR, Choi et al. 2021).** Drop the
  likelihood term; take an unconditional step, then project the iterate onto `{x : A x = y}` to enforce
  data consistency by fiat. Clean and strong when noise is zero. Gap: with noisy `y` the projection
  forces the sample to reproduce the noise (the transpose amplifies it), and projection presupposes a
  linear, easily-projectable `A` — nonlinear operators are out.
- **Spectral-domain solvers — SNIPS / DDRM (Kawar et al. 2021; 2022).** Diagonalize `A` by its SVD and
  handle the measurement-domain Gaussian noise in closed form in the spectral basis. Genuinely
  noise-robust. Gap: needs an explicit, cheap SVD of the forward operator — infeasible for a PDE
  scattering solver, a sparse-interferometry operator, or any general nonlinear `A`.
- **Tweedie's formula (Robbins 1956; Stein 1981; Efron 2011).** The MMSE estimate `E[x_0 | x_t]` is a
  closed-form function of the score, so the denoiser hands back an in-the-loop clean estimate `x_0_hat`
  at any noise level for free. This is the lever every method below pulls on; it is exactly what the
  harness's `net(x, sigma)` returns. Gap: it is only the posterior *mean*, not the posterior.

## The fixed substrate

A pretrained EDM-style diffusion stack is frozen and must not be touched. It provides:

- `net(x, sigma)` → the Tweedie denoised estimate `E[x_0 | x_t]` (one network call); `net.img_channels`,
  `net.img_resolution` give the signal shape.
- `forward_op.forward(x)` → `A(x)`; `forward_op.gradient(x, y, return_loss=True)` → the pair
  `(∇_x ||A(x) - y||², ||A(x) - y||²)`; `forward_op.loss(x, y)` → `||A(x) - y||²` per batch element;
  `forward_op.device`.
- `Scheduler(num_steps, schedule, timestep, scaling, sigma_max, sigma_min, ...)` → the noise schedule,
  exposing `.sigma_steps`, `.factor_steps`, `.scaling_factor`, `.scaling_steps`, `.num_steps`,
  `.sigma_max`. The VP choice (`schedule='vp', timestep='vp', scaling='vp'`) is the default.
- `DiffusionSampler(scheduler).sample(model, x_start, SDE=False)` → unconditional (prior-only) sampling
  from `x_start` down a schedule.

The pretrained denoiser, the forward-operator definitions, and the three evaluation problems are fixed.

## The editable interface

Exactly one region is editable — the `Custom` class in `algo/custom.py`. Every method on the ladder is
a fill of this same contract: `__init__(net, forward_op, ...)` builds the scheduler and stores the
hyperparameters, and `inference(observation, num_samples=1)` returns the reconstruction
`x_recon` of shape `(num_samples, C, H, W)`. A small set of per-problem hyperparameters is keyed off the
`ENV` environment variable (`inv-scatter`, `blackhole`, `inpainting`), because the same conditioning
rule needs a different guidance strength on a clean linear operator than on a noisy sparse one.

The starting point is the scaffold default: **return random noise** (no conditioning at all). Each method
replaces this `Custom` class with its own conditioning rule.

```python
import torch
from tqdm import tqdm
from algo.base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler
import numpy as np


class Custom(Algo):
    """Custom algorithm for solving inverse problems with diffusion priors.

    Available utilities:
        - self.net(x, sigma): denoised estimate E[x_0 | x_t] (Tweedie); .img_channels, .img_resolution.
        - self.forward_op.forward(x): A(x).
        - self.forward_op.gradient(x, y, return_loss=True): (grad ||A(x)-y||^2, loss).
        - self.forward_op.loss(x, y): ||A(x)-y||^2, shape (batch,).
        - self.forward_op.device.
        - Scheduler(num_steps, schedule, timestep, scaling, sigma_max, sigma_min, ...):
              .sigma_steps, .factor_steps, .scaling_factor, .scaling_steps, .num_steps.
        - DiffusionSampler(scheduler).sample(model, x_start, SDE=False): unconditional sampling.
    """

    def __init__(self, net, forward_op,
                 diffusion_scheduler_config=None,
                 guidance_scale=10.0,
                 sde=True,
                 num_optim_steps=1000,
                 observation_weight=1.0,
                 base_lambda=0.25,
                 base_lr=0.5,
                 num_mc_samples=10,
                 **kwargs):
        super(Custom, self).__init__(net, forward_op)
        # Default: no algorithm. Store nothing.
        pass

    def inference(self, observation, num_samples=1, **kwargs):
        """Given y, return x with A(x) ~ y. Default: random noise (no conditioning)."""
        device = self.forward_op.device
        x = torch.randn(num_samples, self.net.img_channels,
                        self.net.img_resolution, self.net.img_resolution,
                        device=device)
        return x
```

## Evaluation settings

Three scientific inverse problems, each keyed by `ENV`, single seed `42`:

1. **Inverse scattering** (`inv-scatter`, optical tomography) — recover permittivity from scattered EM
   fields through a differentiable scattering forward model. Metrics: PSNR, SSIM.
2. **Black hole imaging** (`blackhole`, radio astronomy) — reconstruct an image from sparse EHT
   interferometric observations. Metrics: PSNR, blur-PSNR (f=15), closure-phase chi-squared (`cp_chi2`),
   closure-amplitude chi-squared (`camp_chi2`).
3. **FFHQ-256 inpainting** (`inpainting`, computer vision) — recover a face from a box-masked observation
   with additive Gaussian noise (σ=0.05); the forward operator is a fixed pixel-wise mask. Metrics:
   PSNR, SSIM, LPIPS.

Higher PSNR / SSIM / blur-PSNR is better; lower LPIPS and lower chi-squared are better.
