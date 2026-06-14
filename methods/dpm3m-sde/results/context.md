# Context: Fast Guided Sampling of Diffusion Models in a Tight Step Budget

## Research question

Diffusion probabilistic models produce excellent images, and the technique that makes them
competitive with — and often better than — GANs on fidelity and prompt alignment is *guided
sampling*: at sampling time the network output is steered by a classifier gradient or by a
classifier-free combination of a conditional and an unconditional prediction, with a scalar
*guidance scale* `s` controlling how hard the steering is. Large `s` is what gives sharp,
prompt-faithful, photorealistic images, so practitioners use it routinely.

Sampling means reversing a noising process by running the network many times in sequence. The
dominant guided sampler, DDIM, is reliable but needs on the order of 100 to 250 sequential
network evaluations to converge — far too slow for interactive text-to-image use. Dedicated
high-order ODE solvers cut *unconditional* sampling to 10–20 evaluations, so the obvious move is
to use them for guided sampling too. The precise problem: given an already-trained diffusion
model (which we may not retrain), generate high-quality *guided* samples in as few sequential
network evaluations as possible — on the order of 15–20 — and the solver must be **training-free**
(plug onto any pre-trained model), must not produce worse images than the slow DDIM it replaces,
and must remain well-behaved at the large guidance scales that practitioners actually use.

## Background

**The forward process and noise schedule.** A diffusion model defines a forward process that
gradually adds Gaussian noise to data `x_0`. At time `t`, `q(x_t | x_0) = N(x_t | alpha_t x_0,
sigma_t^2 I)`, i.e. `x_t = alpha_t x_0 + sigma_t eps`, `eps ~ N(0, I)`, where the schedule
`(alpha_t, sigma_t)` is chosen so the signal-to-noise ratio `alpha_t^2 / sigma_t^2` is strictly
decreasing in `t` (Kingma et al. 2021). Two conventions coexist: variance-preserving (VP), where
`alpha_t^2 + sigma_t^2 = 1`, and variance-exploding (VE), where one sets `alpha = 1` and lets a
single noise level `sigma` carry the schedule (Karras et al. 2022). Because the SNR is monotone in
`t`, the quantity `lambda_t = log(alpha_t / sigma_t)` — half the log-SNR — is a strictly decreasing,
invertible function of `t`; in the VE convention with `alpha = 1` it is simply `lambda = -log sigma`.

**Two model parameterizations.** The same network can be read two ways (Kingma et al. 2021). The
*noise-prediction* model `eps_theta(x_t, t)` predicts the added noise. The *data-prediction* model
`x_theta(x_t, t)` predicts the clean datum, and the two are related by
`x_theta = (x_t - sigma_t eps_theta) / alpha_t`. The denoising objective trains either form.

**Sampling as solving a differential equation.** Song et al. (2020) showed the forward process has
a reverse-time stochastic differential equation whose only unknown is the score, and an equivalent
deterministic *probability-flow ODE* sharing the same marginals at every `t`:

```
reverse SDE:  dx_t = [ f(t) x_t + (g^2(t)/sigma_t) eps_theta(x_t, t) ] dt + g(t) dw_bar_t
PF-ODE:       dx_t/dt = f(t) x_t + (g^2(t)/(2 sigma_t)) eps_theta(x_t, t),
```

solved from `t = T` down to `t = 0`, with `f(t) = d log alpha_t / dt` and
`g^2(t) = d sigma_t^2/dt - 2 (d log alpha_t/dt) sigma_t^2`. Discretizing the SDE is intrinsically
costly: each step injects fresh Brownian noise and the step size is capped by that randomness
(Kloeden & Platen 1992), so SDE samplers need many steps; the ODE has no noise term and in principle
tolerates much larger steps.

**The exponential-integrator structure of the ODE.** The PF-ODE right-hand side is *semi-linear*:
a linear term `f(t) x` welded to the nonlinear network term. Variation-of-constants solves the
linear part exactly, and changing the integration variable from `t` to `lambda` collapses all the
schedule coefficients into a single analytic exponential, leaving only an exponentially-weighted
integral of the network (Lu et al. 2022, building on the exponential-integrator / exponential
Runge–Kutta line, Hochbruck & Ostermann 2005, 2010). Concretely, for the noise-prediction model,
between times `s` and `t`,

```
x_t = (alpha_t / alpha_s) x_s - alpha_t * integral_{lambda_s}^{lambda_t} e^{-lambda} eps_theta(x_lambda, lambda) d lambda.
```

The exponential-integrator people define the functions `phi_0(z) = e^z`,
`phi_{k+1}(z) = (phi_k(z) - phi_k(0)) / z`, `phi_k(0) = 1/k!`, which arise from integrating the
Taylor monomials of the network against the exponential weight; these are the standard building
blocks for high-order solvers of semi-linear ODEs.

**Guided-sampling instability — the diagnostic findings.** Two observations about *existing*
high-order solvers, made with the pre-trained guided models, set up the problem:

1. *Large guidance scales destabilize high-order solvers.* For a large guidance scale (e.g.
   `s = 8.0`) on a guided ImageNet 256x256 model at a tight budget (~15 evaluations), the existing
   high-order ODE solvers built on the noise-prediction model — DPM-Solver of order 2 and 3, DEIS,
   PNDM — produce visibly degraded images, *worse* than first-order DDIM, and the degradation grows
   with the solver order. Guidance multiplies both the model output and its derivatives by a large
   factor; high-order solvers lean on those (high-order) derivatives, so the amplification shrinks
   their convergence radius and they would need far smaller steps to converge, which the tight
   budget forbids.

2. *Train-test mismatch.* Image data lives in a bounded interval (`[-1, 1]`). A large guidance scale
   pushes the conditional noise prediction away from the true noise, so the converged clean image
   `x_0` falls outside the bound and renders as saturated, unnatural output (Saharia et al. 2022).
   Saharia et al. observed that *thresholding* — clipping the predicted clean image back into the
   bound at each step — counteracts this, but a clip can only be applied to a quantity that is itself
   an estimate of the clean image.

**Stochasticity as error correction.** Separately, Karras et al. (2022) reported that purely
deterministic sampling can yield *worse* perceptual quality than sampling that re-injects a
controlled amount of noise at each step: the added randomness, removed by the next denoising step,
acts like a Langevin correction that pushes the trajectory back toward the data manifold and
cancels accumulated discretization error. This makes "how much noise to re-inject per step" a knob
worth having rather than something to eliminate.

## Baselines

**DDIM (Song et al. 2020).** A deterministic, non-Markovian sampler; later shown (Salimans & Ho
2022; Lu et al. 2022) to be exactly the first-order discretization of the diffusion ODE. Its general
form carries a stochasticity parameter `eta >= 0`,

```
x_{t_i} = alpha_{t_i} x_theta(...) + sqrt(sigma_{t_i}^2 - eta^2) eps_theta(...) + eta * z,   z ~ N(0, I),
```

with `eta = 0` deterministic and larger `eta` injecting more noise. Reliable and, importantly,
*stable* under large guidance scales. **Gap:** first order only; needs ~100–250 steps; offers no
route to higher order and no high-order error analysis.

**DPM-Solver (Lu et al. 2022).** The high-order exponential-integrator solver built on the
noise-prediction ODE solution above: Taylor-expand `eps_theta` in `lambda`, integrate against
`e^{-lambda}` term by term to get an order-`k` method `DPM-Solver-k` (k network calls per step,
singlestep), with DDIM (`eta = 0`) the first-order case. Reaches ~10–20 steps for *unconditional*
sampling. **Gap:** it is formulated on the noise model `eps_theta`; for *guided* sampling at large
scale it is one of the unstable solvers in finding (1) above, and because it never forms an estimate
of the clean image, thresholding cannot be applied to keep its output bounded (finding (2)).

**DEIS (Zhang & Chen 2022).** Another exponential-integrator solver, *multistep* in flavor (it
reuses past network evaluations, Adams–Bashforth style, instead of inserting fresh intermediate
evaluations within a step), Taylor-expanding `eps_theta` in `t`. Strong unconditionally. **Gap:**
same noise-model formulation; same guided-sampling instability at large scale.

**Ancestral / SDE samplers (Ho et al. 2020; Song et al. 2020).** Discretizations of the reverse
SDE. Excellent quality but slow — randomness caps the step size, so they need many steps to converge.

**Thresholding (Ho et al. 2020; Saharia et al. 2022).** A fix for the bounded-data problem, not a
solver: clip the predicted clean image (static clipping to `[-1, 1]`, or dynamic clipping by a
per-step percentile) so out-of-range pixels are pulled back in. It needs a clean-image estimate to
clip, so it composes only with a solver that produces one.

**Karras / EDM schedule (Karras et al. 2022).** Not a solver but the natural time grid: place the
sampling noise levels as `sigma_i = (sigma_max^{1/rho} + (i/(N-1))(sigma_min^{1/rho} -
sigma_max^{1/rho}))^rho`, with `rho = 7`, then append `sigma = 0`. The power warp concentrates more
of the budget at low noise, where the per-step truncation error is largest.

## Evaluation settings

The natural yardsticks are guided image-generation benchmarks on which the pre-trained models were
already evaluated. Class-conditional ImageNet 256x256 with classifier guidance over a range of
guidance scales (e.g. 0 through 8). Latent-space and pixel-space text-to-image models (Stable
Diffusion / latent diffusion; pixel-space cascaded models) with classifier-free guidance at the
guidance scales practitioners use (e.g. 7.5). Sample quality is measured by Fréchet Inception
Distance (FID, lower is better) against a reference set, and by prompt alignment via a CLIP score
(higher is better). The cost axis — the whole point — is the number of sequential network
evaluations (NFE); a fast guided sampler is judged by the FID it reaches at a fixed small NFE,
especially in the 10–20 range. The model weights, the prompt set, the guidance scale, the NFE
budget, and the metric computation are all held fixed across solvers; only the per-step update and
its time/noise grid vary.

## Code framework

Before any dedicated solver, what exists is a generic sampling harness: a schedule object that
knows `alpha_t`, `sigma_t` and the derived `lambda_t`; a wrapper that turns the trained network
into a callable returning its prediction at `(x, sigma)`; a routine that lays out a decreasing
sequence of noise levels; and a loop that marches the latent from pure noise down to a clean image,
calling the network once per step. The contribution lives entirely inside the per-step update — how
to advance the latent from one noise level to the next given the current network output (and
whatever past information we choose to keep) — and inside how the noise levels are spaced.

```python
import torch


class Schedule:
    """Known: the noise schedule and quantities derived from it."""
    def alpha(self, t): ...
    def sigma(self, t): ...
    # half log-SNR lambda_t = log(alpha_t) - log(sigma_t), strictly decreasing in t,
    # and its inverse. (In the alpha = 1 convention, lambda = -log sigma.)
    def lam(self, t): ...
    def inverse_lam(self, lam): ...


def get_noise_levels(n, sigma_min, sigma_max, device):
    """Known: a decreasing sequence of n+1 noise levels from sigma_max down to 0.
    The right spacing (uniform in sigma? in log-sigma? in some warped coordinate?) is
    part of what must be decided."""
    raise NotImplementedError  # TODO


def wrap_model(net, schedule):
    """Known: turn the trained network into a callable prediction at (x, sigma),
    optionally combining a conditional and an unconditional pass for guidance."""
    def predict(x, sigma):
        ...  # return the network's prediction at this noise level
    return predict


class Sampler:
    def __init__(self, predict, schedule):
        self.predict = predict
        self.ns = schedule

    def step(self, x, sigma, sigma_next, history):
        # Advance the latent from noise level sigma to sigma_next (sigma_next < sigma),
        # given the current network prediction and any retained past information.
        # This is the slot the method fills in.
        raise NotImplementedError  # TODO

    @torch.no_grad()
    def sample(self, x, sigmas):
        history = {}
        for i in range(len(sigmas) - 1):
            x = self.step(x, sigmas[i], sigmas[i + 1], history)
        return x
```
