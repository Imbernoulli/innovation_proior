# Context: Fast Guided Diffusion Sampling with High-Order ODE Solvers

## Research question

Diffusion models generate images by reversing a noising process, running a trained network once per
step down a sequence of noise levels. For interactive text-to-image use the budget is tight —
on the order of 5 to 20 network evaluations — and the images must stay sharp and prompt-faithful, which
in practice means *guided* sampling at a large guidance scale. Fast high-order ODE solvers
(DPM-Solver, DPM-Solver++, DEIS) already cut unconditional sampling to ~10–20 steps, and DPM-Solver++
restored stability under guidance by moving to the data-prediction parameterization. Given an
already-trained model (no retraining), how can one build a training-free sampler that achieves high
accuracy at small NFE — especially in the extreme few-step regime (5–10 steps)?

## Background

**Forward process, schedule, half-log-SNR.** A diffusion model defines `q(x_t|x_0) = N(alpha_t x_0,
sigma_t^2 I)`, i.e. `x_t = alpha_t x_0 + sigma_t eps`, with the signal-to-noise ratio
`alpha_t^2/sigma_t^2` strictly decreasing in `t` (Kingma et al. 2021). The half-log-SNR
`lambda_t = log(alpha_t/sigma_t)` is therefore strictly monotone and invertible; one step of any
exponential-integrator solver is parameterized by `h = lambda_t - lambda_s` (the change in `lambda`
across the step).

**Two parameterizations.** The same network reads as a noise predictor `eps_theta(x_t,t)` or a
data predictor `x_theta(x_t,t) = (x_t - sigma_t eps_theta)/alpha_t` (Kingma et al. 2021). The
data-prediction form stays stable under large classifier-free guidance and exposes a clean-image
estimate.

**Exponential-integrator solution.** The diffusion ODE is semi-linear; variation of constants plus the
change of variable to `lambda` gives, in the data-prediction form, the exact step
`x_t = (sigma_t/sigma_s) x_s + sigma_t * integral_{lambda_s}^{lambda_t} e^{lambda} x_theta(lambda) d
lambda` (Lu et al. 2022). The exponential-integrator `phi` functions
`phi_1(z) = (e^z - 1)/z`, `phi_{k+1}(z) = (phi_k(z) - phi_k(0))/z`, `phi_k(0) = 1/k!`, arise from
integrating the Taylor monomials of `x_theta` against the exponential weight (Hochbruck & Ostermann
2010). Existing multistep solvers (DPM-Solver++(kM), DEIS) approximate the derivatives of `x_theta`
from past network outputs and weight them by these `phi`s.

**Classical predictor-corrector pairs in ODE numerics.** A classical technique in ODE numerics is a
*predictor-corrector* pair (Adams–Moulton on top of Adams–Bashforth): take a predictor step, evaluate
the right-hand side at the predicted point, then *correct* using that new evaluation.

**Guidance stability and schedule.** Large guidance scales amplify the model's derivatives; latent-space
models have no `[-1,1]` bound to threshold; a power-law (Karras/EDM, `rho=7`) or uniform-`lambda` time
grid concentrates steps where truncation error is concentrated.

## Baselines

**DDIM (Song et al. 2020).** First-order discretization of the diffusion ODE; reliable and stable
under guidance.

**DPM-Solver / DPM-Solver++ (Lu et al. 2022).** High-order exponential-integrator solvers, the `++`
family on the data prediction for guidance stability; singlestep (kS) and multistep (kM) variants.

**DEIS (Zhang & Chen 2022).** Multistep exponential-integrator solver on the noise prediction.

**Karras/EDM schedule (Karras et al. 2022).** The time grid, not a solver: `sigma_i = (sigma_max^{1/rho}
+ (i/(N-1))(sigma_min^{1/rho} - sigma_max^{1/rho}))^rho`, `rho=7`, then `sigma=0`.

## Evaluation settings

Guided image-generation benchmarks at fixed small NFE. Class-conditional ImageNet 256x256 with
classifier guidance; latent text-to-image (Stable Diffusion / SDXL) with classifier-free guidance at
the scales practitioners use (e.g. 7.5). Quality by FID (lower better) against a reference set and CLIP
score (higher better); cost by NFE, judged especially in the 5–20 range. Weights, prompts, guidance
scale, NFE budget, and metric computation are held fixed across solvers; only the per-step update and
its grid vary.

## Code framework

A generic multistep sampling harness: a schedule object knowing `alpha_t, sigma_t, lambda_t`; a wrapper
turning the network into a `(x, sigma) -> x_theta` prediction (combining conditional and unconditional
passes for guidance); a routine laying out the decreasing noise levels; and a loop marching the latent
down, calling the network once per step and keeping a short history of past model outputs and step
sizes. The contribution lives in the per-step update and in how it uses the history of past model
outputs and step sizes.

```python
import torch


class Schedule:
    """Known: the noise schedule and quantities derived from it."""
    def alpha(self, t): ...
    def sigma(self, t): ...
    def lam(self, t): ...           # half log-SNR, strictly decreasing in t
    def inverse_lam(self, lam): ...


def get_noise_levels(n, sigma_min, sigma_max, device):
    """Known: a decreasing sequence of n+1 noise levels down to 0 (spacing TBD)."""
    raise NotImplementedError  # TODO


def wrap_model(net, schedule):
    """Known: (x, sigma) -> data prediction x_theta, combining cond/uncond for guidance."""
    def predict(x, sigma):
        ...
    return predict


class Sampler:
    def __init__(self, predict, schedule):
        self.predict = predict
        self.ns = schedule

    def update_step(self, x, sigma, sigma_next, history, order):
        # advance the latent using history of past model outputs
        raise NotImplementedError  # TODO

    @torch.no_grad()
    def sample(self, x, sigmas):
        history = {}
        for i in range(len(sigmas) - 1):
            x = self.update_step(x, sigmas[i], sigmas[i + 1], history, order=...)
        return x
```
