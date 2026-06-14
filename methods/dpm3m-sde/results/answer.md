# DPM-Solver++(3M) SDE, distilled

DPM-Solver++(3M) SDE is a training-free, third-order **multistep stochastic** exponential-integrator
sampler for diffusion models, built on the **data-prediction** (clean-image, `x0`) parameterization,
with a single knob `eta` that interpolates between the deterministic probability-flow ODE and the
full reverse SDE. It is the high-order, stochastic descendant of DDIM designed to make *guided*
sampling fast (15–20 network evaluations) without the instability that noise-prediction high-order
solvers suffer at large guidance scales.

## Problem it solves

Fast guided sampling of a pre-trained diffusion model in a tight evaluation budget. Existing
high-order ODE solvers (built on the noise model `eps_theta`) become *worse* than first-order DDIM at
the large guidance scales practitioners use, for two reasons: (1) guidance amplifies the model's
`lambda`-derivatives, shrinking the convergence radius the high-order terms need; (2) the converged
clean image leaves the data range (`[-1,1]`), and a noise-prediction solver carries no clean-image
estimate to threshold back into range.

## Key idea

Re-derive the exponential-integrator solver on the **data-prediction face**. The diffusion ODE in
`x_theta := (x_t - sigma_t eps_theta)/alpha_t` is semi-linear; variation of constants plus the change
of variable `lambda = log(alpha_t/sigma_t)` (half log-SNR; `lambda = -log sigma` when `alpha = 1`)
gives the exact solution

```
x_t = (sigma_t/sigma_s) x_s + sigma_t * integral_{lambda_s}^{lambda_t} e^{lambda} x_theta(x_lambda, lambda) d lambda.
```

This exactly-carries `sigma_t/sigma_s` and only ever approximates an integral of `x_theta` — the
clean-image estimate — so clipping or dynamic thresholding can be applied directly to the model
quantity that needs to stay in the data range.

For the reverse **SDE** (in `lambda`, VP):

```
dx = [-(1 + alpha^2) x + 2 alpha x_theta] d lambda + sqrt(2) sigma d w_lambda,
x_t = (sigma_t/sigma_s) e^{-h} x_s + 2 alpha_t * integral e^{-2(lambda_t - lambda)} x_theta d lambda
        + sqrt(2) sigma_t * integral e^{-(lambda_t - lambda)} d w_lambda,   h = lambda_t - lambda_s.
```

The Itô integral is Gaussian with variance `(1 - e^{-2h})/2`, so the noise term is
`sigma_t sqrt(1 - e^{-2h}) z`, `z ~ N(0,I)`. First-order (`x_theta` held constant):

```
x_t = (sigma_t/sigma_s) e^{-h} x_s + alpha_t (1 - e^{-2h}) x_theta(x_s,s) + sigma_t sqrt(1 - e^{-2h}) z,
```

which equals DDIM with `eta_DDIM = sigma_t sqrt(1 - e^{-2h})`.

**The `eta` knob.** Introduce `h_eta = h (eta + 1)` so the signal contracts by `e^{-h_eta}` and the
clean prediction takes the complement `1 - e^{-h_eta}` (a convex split), with injected noise standard
deviation `sigma_next sqrt(1 - e^{-2 h eta})`:

- `eta = 0`: `h_eta = h`, no noise → deterministic ODE step.
- `eta = 1`: `h_eta = 2h`, noise `sqrt(1 - e^{-2h})` → full SDE step (stochastic DDIM at order 1).
- `eta > 1`: extra Langevin stochasticity for robustness under guidance / large steps.

**Multistep + order.** One network call per step, reusing the last two `x_theta` values and step
sizes. With `lambda`-spacings scaled to the current step (`r0 = h_1/h`, `r1 = h_2/h`), a quadratic
Newton divided-difference recovers the first and second `lambda`-derivatives of `x_theta`, weighted by
the exponential-integrator functions

```
phi_2(h_eta) = (e^{-h_eta} - 1)/h_eta + 1   (~ h_eta/2,  first-derivative weight),
phi_3(h_eta) = phi_2(h_eta)/h_eta - 0.5     (~ -h_eta/6, second-derivative weight, negative),
```

giving the correction `+ phi_2 d1 - phi_3 d2`. The `phi_2` first-derivative term matches the exact
integral coefficient; the `phi_3` curvature term is a practical correction (local error ~ `O(h^3)`),
in the same spirit as the second-order solver's simplified `phi`-coefficient — a robust higher-order
step rather than an exact `O(h^4)` scheme. Orders drop out by how much history is available: no
history → first order; one past value → second; two → third.

**Why these choices.** Data-prediction (over noise-prediction): clippable clean predictions,
threshold composition, smaller error constant. Multistep (over singlestep): at fixed budget `N`,
`M = N` steps vs `N/k`, so smaller `h` keeps the solver inside the guidance-narrowed convergence
radius, and expensive network evaluations are reused. SDE/`eta > 0`: Langevin re-noising cancels
accumulated discretization error, often beating the deterministic ODE perceptually. Karras `rho = 7`
schedule: concentrate steps at low `sigma` where truncation error lives. `expm1`: stable
small-argument exponentials. Brownian-tree noise: reproducible, correctly-scaled increments.

## Final algorithm (per step, `denoised = x_theta(x, sigma)`)

```
for i = 0 .. N-1:
    denoised = model(x, sigma_i)
    if sigma_{i+1} == 0:
        x = denoised                                     # final step at sigma -> 0
    else:
        t, s = -log sigma_i, -log sigma_{i+1};  h = s - t;  h_eta = h (eta + 1)
        x = e^{-h_eta} x + (1 - e^{-h_eta}) denoised      # constant data-prediction step
        if have two past values:
            r0 = h_1/h;  r1 = h_2/h
            d1_0 = (denoised - denoised_1)/r0;  d1_1 = (denoised_1 - denoised_2)/r1
            d1 = d1_0 + (d1_0 - d1_1) r0/(r0+r1);  d2 = (d1_0 - d1_1)/(r0+r1)
            phi_2 = (e^{-h_eta}-1)/h_eta + 1;  phi_3 = phi_2/h_eta - 0.5
            x += phi_2 d1 - phi_3 d2
        elif have one past value:
            r = h_1/h;  d = (denoised - denoised_1)/r
            x += ((e^{-h_eta}-1)/h_eta + 1) d
        if eta:  x += noise * sigma_{i+1} * sqrt(1 - e^{-2 h eta}) * s_noise
    shift history: denoised_1, denoised_2 = denoised, denoised_1;  h_1, h_2 = h, h_1
return x
```

## Working code

Faithful to the standard k-diffusion implementation. `model(x, sigma)` returns the clean-image
estimate `x_theta`; the VE convention sets `alpha = 1`, so `lambda = -log sigma`. The third-order
case intentionally scales the older interval as `r1 = h_2 / h` and applies the curvature correction
as `x = x + phi_2 * d1 - phi_3 * d2`.

```python
import torch
from tqdm.auto import trange


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Karras et al. (2022) power schedule: more steps where sigma is small."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros([1])]).to(device)        # append final sigma = 0


@torch.no_grad()
def sample_dpmpp_3m_sde(model, x, sigmas, extra_args=None, callback=None,
                        disable=None, eta=1., s_noise=1., noise_sampler=None):
    """DPM-Solver++(3M) SDE: third-order multistep stochastic data-prediction solver."""
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = (BrownianTreeNoiseSampler(x, sigma_min, sigma_max)
                     if noise_sampler is None else noise_sampler)
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None      # last two data-predictions
    h_1, h_2 = None, None                    # last two lambda-step-sizes

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)            # x_theta
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i],
                      'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] == 0:
            x = denoised                                               # final denoising step
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()             # lambda = -log sigma
            h = s - t
            h_eta = h * (eta + 1)                                      # eta: ODE(0) <-> SDE(1)

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:                                        # third order
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:                                      # second order
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:                                                    # Langevin renoise
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] \
                    * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x
```

The `BrownianTreeNoiseSampler` wraps a `torchsde.BrownianTree` so the injected noise is
seed-reproducible and carries correctly-scaled Brownian increments (each increment divided by
`sqrt(|delta t|)`) across the noise-level grid.

## Relation to prior methods

- **DDIM** is the first-order case: deterministic (`eta = 0`) on the data face, or stochastic DDIM
  with `eta_DDIM = sigma_t sqrt(1 - e^{-2h})` at `eta = 1`.
- **DPM-Solver** is the same exponential-integrator construction on the *noise* face (`eps_theta`,
  singlestep); the data face here exposes a clippable clean prediction and yields a smaller error
  constant (an extra `e^{-r h} < 1` factor on the first-derivative term when rewritten in `eps`),
  which is what restores stability under large guidance.
- **DEIS** is a noise-face multistep solver; same instability at large guidance.
- The **deterministic** DPM-Solver++(2M)/(3M) are the `eta = 0` special cases; the SDE variants add the
  `eta`-controlled Langevin renoising. The second-order multistep solver (DPM-Solver++(2M) SDE) is the
  case with one past value; this is its third-order multistep extension.
- The time grid is the **Karras / EDM** power schedule (`rho = 7`).
