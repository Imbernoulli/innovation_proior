# DPM-Solver++(3M) SDE, distilled

DPM-Solver++(3M) SDE is a training-free, **three-point multistep stochastic** sampler for
pre-trained diffusion models, built to make guided sampling fast at a tight network-call budget without
the instability that wrecks naive high-order solvers at large guidance scales. It is an exponential
integrator on the **data-prediction** (`x_theta`, clean-image) face of the diffusion ODE/SDE, with a
single `eta` knob interpolating the per-step renoising rate from the deterministic ODE to the full
SDE and beyond, marched on a Karras `sigma` schedule. Its 3M correction is a Newton divided-difference
estimate of the model output's first two `lambda`-derivatives, assembled so the finite differences are
dimensionally consistent and the curvature term carries the canonical sign.

## Problem it solves

Generate high-quality guided images from an already-trained diffusion model in as few sequential
network evaluations as possible, training-free, at the large guidance scales practitioners use. Two
pathologies break the obvious "use a fast high-order ODE solver" approach: (1) guidance
amplifies the model output *and its derivatives*, shrinking a high-order solver's convergence radius
so that at a fixed small budget it becomes *worse* than first-order DDIM (worse as order rises); and
(2) on bounded image data `[-1, 1]`, large guidance pushes the converged clean image out of range
(saturated output) — fixable by thresholding, but only a solver that *holds* a clean-image estimate
can be thresholded.

## Key ideas

- **Solve on the data-prediction face.** The same network reads as `eps_theta` or as
  `x_theta = (x_t - sigma_t eps_theta)/alpha_t`. Re-deriving the exponential-integrator solver on
  `x_theta` gives the exact solution
  `x_t = (sigma_t/sigma_s) x_s + sigma_t * integral_{lambda_s}^{lambda_t} e^{lambda} x_theta dl`,
  whose every approximation lives in an integral of the *bounded, clippable* clean prediction. The
  per-step update is a **convex combination** of the latent and that prediction (weights
  `e^{-h_eta}` and `1 - e^{-h_eta}` sum to 1), which is the structural source of boundedness.
- **Multistep, not singlestep.** At a fixed budget `N`, a singlestep order-`k` method affords only
  `N/k` steps (big steps), while a multistep method (1 call/step + reuse of past predictions)
  affords `N` steps (steps `k`× smaller). Smaller `h` shrinks `O(h^k)` error and keeps the solver
  inside the convergence radius that guidance narrows.
- **Stochastic, with one knob `eta`.** Solving the diffusion *SDE* (drift/noise coefficient twice
  the ODE's) with the same machinery adds a Langevin self-correction: re-injected noise, removed by
  the next denoising step, cancels accumulated discretization error. `eta` interpolates the
  renoising rate: `h_eta = h(eta + 1)`; `eta = 0` recovers the deterministic ODE, `eta = 1` the full
  SDE, `eta > 1` extra robustness.
- **Karras `sigma` schedule (rho = 7).** Concentrate steps at low noise where truncation error
  dominates; this also makes the `lambda`-spacings unequal — which the multistep divided differences
  must handle correctly.

## The half-log-SNR variable and the `eta` step

With VE convention `alpha = 1`, the noise level `sigma` carries the schedule and the half-log-SNR is
`lambda = -log sigma`. For a step `sigma_i -> sigma_{i+1}` (decreasing), set
`t = -log sigma_i`, `s = -log sigma_{i+1}`, `h = s - t > 0`, and `h_eta = h(eta + 1)`. The constant
(first-order) data-prediction SDE step is the convex combination plus a renoise:

```
x  <-  exp(-h_eta) * x + (1 - exp(-h_eta)) * denoised          # denoised = x_theta
if eta:  x  <-  x + z * sigma_{i+1} * sqrt(1 - exp(-2 h eta))  # z Brownian-tree noise
```

Endpoints: `eta = 0` gives `e^{-h} x + (1 - e^{-h}) denoised`, no noise (deterministic ODE);
`eta = 1` gives `e^{-2h} x + (1 - e^{-2h}) denoised + sigma_{i+1} sqrt(1 - e^{-2h}) z` (full SDE,
matching the first-order stochastic step `x_t = (sigma_t/sigma_s) e^{-h} x_s + alpha_t(1 - e^{-2h})
x_theta + sigma_t sqrt(1 - e^{-2h}) z`, which equals DDIM with `eta_DDIM = sigma_t sqrt(1 - e^{-2h})`).
The renoise variance `1 - e^{-2 h eta}` restores exactly the marginal variance removed by the extra
SDE contraction `e^{-h eta}` (amplitude) = `e^{-2 h eta}` (variance) beyond the ODE.

## The multistep correction (the divided-difference bookkeeping)

Taylor-expanding `x_theta` in `lambda` and integrating the first monomial against the exponential
weight gives `phi_2`; the canonical 3M curvature helper is `phi_3`, also evaluated at `h_eta`:

```
phi_2 = (exp(-h_eta) - 1)/h_eta + 1      # ~ h_eta/2  (>0)   coefficient on h * D_1
phi_3 = phi_2/h_eta - 0.5                # ~ -h_eta/6 (<0)   helper for the curvature term
```

The needed quantities are `h * x_theta^{(1)}(lambda_s)` and `(h^2/2) * x_theta^{(2)}(lambda_s)`,
estimated by a Newton divided difference from three past data-predictions at unequal `lambda`-spacings
`h_1` (one step back) and `h_2` (the step before that). **Both** backward differences are scaled to
the *current* step `h`, so they share a common unit (`h * D_1`) and line up with the `phi` weights:

```
r0 = h_1 / h
r1 = h_2 / h                              # NOT h_2/h_1 -- keeps the common unit
d1_0 = (denoised   - denoised_1) / r0     # ~ h * x_theta^(1), near interval
d1_1 = (denoised_1 - denoised_2) / r1     # ~ h * x_theta^(1), next interval back
d1   = d1_0 + (d1_0 - d1_1) * r0/(r0+r1)  # endpoint first-derivative est, = h * D_1
d2   = (d1_0 - d1_1) / (r0 + r1)          # second-difference,            = (h^2/2) * D_2
x  <-  x + phi_2 * d1 - phi_3 * d2        # -phi_3 (>0) applies a positive curvature multiplier
```

Two bookkeeping points are load-bearing and are exactly what makes this the canonical form:

- **`r1 = h_2 / h` (not `h_2 / h_1`).** The `phi` weights are functions of `h_eta`, attached to the
  current step `h`, so every finite difference must be a multiple of `h`. With `r1 = h_2/h`, both
  `d1_0` and `d1_1` are in units of `h * D_1`, so `d1_0 - d1_1` is a clean second difference and
  `d2` recovers `(h^2/2) D_2`. With `r1 = h_2/h_1`, `d1_1` ends up in units of `h_1 * D_1`; even a
  linear model output then produces fake curvature unless `h_1 = h`.
- **`- phi_3 * d2` (not `+ phi_3 * d2`).** The exact Taylor coefficient on the second derivative is
  positive; in this `d2 = (h^2/2)D_2` normalization the fully tuned multiplier would be `2(-phi_3)`.
  Canonical k-diffusion uses the same-sign practical half-weight `-phi_3`. Because `phi_3 < 0`,
  `-phi_3 * d2` applies a positive multiplier to the curvature estimate. Writing `+ phi_3 * d2` flips
  that sign.

**Honest order claim.** `phi_2 * d1` matches the exact first-derivative integral coefficient. The
`-phi_3 * d2` term is a *practical* second-derivative correction (local truncation `~O(h^3)`, not the
`O(h^4)` of a perfectly-tuned third-order multistep) — the same spirit as the second-order SDE
derivation replacing the exact `(e^{-2h} - 1 + 2h)/(2h)` by the simpler `(1 - e^{-2h})/2`. It is not
"provably exactly third order"; it is a stable 3M multistep correction with the correct
scaling and sign, which is what matters under guidance.

The order ramps with history: first step first-order (constant term only); second step second-order
(`x += phi_2 * d` with one past value, `d = (denoised - denoised_1)/r`, `r = h_1/h`); thereafter
the 3M curvature correction is used. The final step (`sigma -> 0`) returns `denoised` (Tweedie clean
prediction), no renoise.

## Working code

Fills the `step`/`sample` slot of the generic sampling harness. `denoised = x_theta`. `expm1` keeps
the `e^x - 1` cancellations honest; `BrownianTreeNoiseSampler` gives reproducible, correctly-scaled
Brownian increments.

```python
import torch
from tqdm.auto import trange


@torch.no_grad()
def sample_dpmpp_3m_sde(model, x, sigmas, extra_args=None, callback=None,
                        disable=None, eta=1., s_noise=1., noise_sampler=None):
    """DPM-Solver++(3M) SDE: three-point multistep stochastic data-prediction
    exponential-integrator sampler. model(x, sigma) returns x_theta ('denoised')."""
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)            # x_theta at this noise level
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] == 0:
            x = denoised                                              # last step: return clean estimate
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()             # lambda = -log sigma (alpha = 1)
            h = s - t
            h_eta = h * (eta + 1)                                     # eta: ODE (0) <-> SDE (1) <-> more

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised   # convex combination

            if h_2 is not None:                                       # 3M curvature correction
                r0 = h_1 / h
                r1 = h_2 / h                                          # both differences in units of h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)            # ~ h * x_theta^(1)
                d2 = (d1_0 - d1_1) / (r0 + r1)                        # ~ (h^2/2) * x_theta^(2)
                phi_2 = h_eta.neg().expm1() / h_eta + 1               # > 0
                phi_3 = phi_2 / h_eta - 0.5                           # < 0
                x = x + phi_2 * d1 - phi_3 * d2                       # -phi_3 (>0) adds curvature estimate
            elif h_1 is not None:                                     # second order
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:                                                  # Langevin renoise
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] \
                    * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1                 # shift history
        h_1, h_2 = h, h_1
    return x


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Karras power schedule: concentrate steps at low sigma; append sigma = 0."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros([1])]).to(device)
```

## Defaults and why

- `eta = 1`: stochastic renoising for Langevin self-correction;
  `eta = 0` recovers the deterministic ODE, `eta = 1` the full SDE, `eta > 1` more robustness under
  guidance and large steps.
- `rho = 7`: Karras power warp concentrating the tight budget at low noise.
- `s_noise = 1`: optional noise-scale multiplier, tunable without touching `eta`.
- 3M correction: uses two retained model outputs to add a curvature estimate when stable; small
  multistep `h` plus the SDE noise self-correction make that correction usable under guidance.

## Relation to prior methods

- **DDIM** is the deterministic first-order case at `eta = 0`; the full-SDE first-order step at
  `eta = 1` is stochastic DDIM with `eta_DDIM = sigma_t sqrt(1 - e^{-2h})`:
  `x_t = (sigma_t/sigma_s) e^{-h} x_s + alpha_t(1 - e^{-2h}) x_theta +
  sigma_t sqrt(1 - e^{-2h}) z`.
- **DPM-Solver** is the singlestep exponential integrator on the *noise* face; this method moves to
  the *data* face (boundedness, smaller error constant via the extra `e^{-rh} < 1` factor),
  multistep (reuse + small steps), and stochastic (the `eta` knob).
- **DEIS** is multistep on the noise face; same data-vs-noise distinction.
- **Karras / EDM** supplies the `sigma` schedule and the deterministic-vs-stochastic insight that
  motivates `eta > 0`.
