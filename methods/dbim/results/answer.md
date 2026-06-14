# Diffusion Bridge Implicit Models (DBIM), distilled

DBIM is a training-free fast sampler for pretrained denoising diffusion bridge models (DDBMs).
It generalizes the bridge's reverse process into a family of *non-Markovian* diffusion bridges,
defined on the discretized sampling timesteps and controlled by a per-step injected standard
deviation `ρ`, while sharing the *same marginals* as the original bridge. For positive `ρ`, the
induced variational objective has the same score-matching minimizer up to timestep weights, so the
pretrained network is reused unchanged; the deterministic sampler is the noiseless endpoint of
that same marginal-preserving construction. The family spans deterministic (`η = 0`, an implicit model with
a clean ODE that admits high-order solvers) to Markovian-stochastic (`η = 1`, DDPM-like). DBIM is
the bridge counterpart of DDIM and is designed for few-step sampling without changing training.

## Problem it solves

Sampling a pretrained diffusion bridge `q(x_t | x_0, x_T) = N(a_t x_T + b_t x_0, c_t² I)` (for
image-to-image translation / restoration with informative endpoint `x_T`) without retraining and
under a tight per-sample budget on denoiser calls (NFE). The bridge coefficients are
`a_t = (α_t/α_T)(SNR_T/SNR_t)`, `b_t = α_t(1 - SNR_T/SNR_t)`,
`c_t^2 = σ_t^2(1 - SNR_T/SNR_t)`. The existing sampler is a generic SDE/ODE discretizer
(DDBM's EDM-Heun hybrid) that takes many network calls.

## Key idea

The denoising-bridge-score-matching loss depends on the model only through the per-time
*marginals* `q(x_t | x_0, x_T)`, not the joint. So replace the joint inference process with a
non-Markovian family that preserves those marginals, indexed by per-step injected std `ρ_n`:

```
q^ρ(x_{t_n} | x_0, x_{t_{n+1}}, x_T)
  = N( a_{t_n} x_T + b_{t_n} x_0 + sqrt(c_{t_n}^2 - ρ_n^2) · (x_{t_{n+1}} - a_{t_{n+1}} x_T - b_{t_{n+1}} x_0)/c_{t_{n+1}},  ρ_n^2 I ),
```
with the boundary `ρ_{N-1} = c_{t_{N-1}}`. The recycled term is the normalized noise of the later
state — `ε̂ = (x_{t_{n+1}} - a_{t_{n+1}} x_T - b_{t_{n+1}} x_0)/c_{t_{n+1}}` — so the update is
"bridge mean at `t_n` + predicted noise direction + fresh noise."

- **Marginal preservation (proved by backward induction).** Base `n = N-1`: `ρ_{N-1} = c_{t_{N-1}}`
  ⇒ recycled coefficient `= 0` ⇒ conditional `= N(a x_T + b x_0, c² I)`. Inductive: a Gaussian
  convolved with a Gaussian; the recycled-direction term averages to zero (since `x_{t_k}` sits
  at its bridge mean in expectation), and the variance bookkeeping gives
  `ρ_{k-1}^2 + (sqrt(c_{k-1}^2 - ρ_{k-1}^2)/c_k)^2 · c_k^2 = c_{k-1}^2`. Holds for every
  admissible `ρ` with the boundary above.
- **Training equivalence (proved via the ELBO).** The variational objective collapses to a
  weighted sum of data-prediction errors `Σ (d_{n-1}^2/2ρ_{n-1}^2)‖x_θ - x_0‖^2`, which converts
  to the score-matching loss `Σ γ(t_n)‖s_θ - ∇log q‖^2` via `‖x_θ - x_0‖^2 = (c^4/b^2)‖s_θ - ∇log q‖^2`,
  `s_θ = -(x_t - a_t x_T - b_t x_θ)/c_t^2`. Different weighting `γ`, same minimizer for positive
  `ρ`; the deterministic endpoint is used as the noiseless sampler in the same family.

## The generative update and the stochasticity dial

Replace `x_0` by the data prediction `x̂_0 = x_θ(x_{t_{n+1}}, t_{n+1}, x_T)`:
```
x_{t_n} = a_{t_n} x_T + b_{t_n} x̂_0 + sqrt(c_{t_n}^2 - ρ_n^2) · ε̂ + ρ_n ε,   ε ~ N(0, I).
ε̂ = (x_{t_{n+1}} - a_{t_{n+1}} x_T - b_{t_{n+1}} x̂_0) / c_{t_{n+1}}.
```
Parameterize `ρ` by `η ∈ [0, 1]`:
```
ρ_n = η · σ_{t_n} sqrt(1 - SNR_{t_{n+1}}/SNR_{t_n}),    SNR_t = α_t^2/σ_t^2.
```
Here `ρ_n` is the injected transition standard deviation; in the implementation, variables named
`rho_s` and `rho_t` are schedule values `σ/α = 1/sqrt(SNR)`, so `alpha_t * rho_t = σ_t`.
- `η = 1` (Markovian boundary): the `x_0`-coefficient of the induced forward process vanishes,
  the `x_T` term cancels, and the update reduces to a DDPM-like ancestral sampler.
- `η = 0`: deterministic implicit model (DBIM proper) — sharp, invertible, few-step.
- Intermediate `η`: non-Markovian, partial stochasticity (a Langevin-style correction that can
  improve quality on diverse tasks).

## Booting noise (the first-step singularity) and the sharp last step

At `η = 0` the first step (`t_{n+1} = T`) divides by `c_T = 0` — the bridge under fixed `x_T` is
genuinely stochastic (`p(x_t | x_T)` is not a Dirac), so determinism there is ill-posed. Using
the Markovian-boundary `ρ_{N-1} = c_{t_{N-1}}` at step one zeros the recycled coefficient and
leaves a single injected Gaussian, the **booting noise**: it places the initial state on the
bridge `x = a x_T + b x̂_0 + c ε`, accounts for the spread of `x_0` given `x_T`, and acts as the
latent variable (fix it to get a deterministic, invertible run for encoding / interpolation). The
fresh noise is also **dropped on the final step** to keep the endpoint sharp.

## Limiting cases

- **DDIM** is the small-`t` limit (`SNR_T/SNR_t → 0`, so `a_t → 0, b_t → α_t, c_t → σ_t`):
  `x_s ≈ (σ_s/σ_t) x_t + σ_s(α_s/σ_s - α_t/σ_t) x̂_0`.
- **Flow matching** is a degenerate Brownian-bridge schedule: `x_s = x_t - (t - s) v_θ`.

## Deterministic ODE and high-order solvers

Dividing the `η = 0` update by `c_t` gives, in the continuous limit, a clean ODE:
```
d(x_t/c_t) = x_T d(a_t/c_t) + x_θ(x_t, t, x_T) d(b_t/c_t),
```
equivalent to the bridge PF-ODE but written on `x_t/c_t`. Expanding gives the semilinear
coefficients `A = c_t'/c_t`, `B_T = a_t' - a_t c_t'/c_t`, and
`B_θ = b_t' - b_t c_t'/c_t`. Variation-of-constants has integrating factor
`exp(∫_s^t A) = c_t/c_s`. In the equations below, `t` is the current larger time and
`s < t` is the next smaller time; the implementation uses the opposite variable names in its
loop. With `λ_t = log(b_t/c_t)` (so `b_t/c_t = sqrt(SNR_t - SNR_T)`, and the closed form is
`λ_t = 0.5 log(SNR_t - SNR_T)`), the exact solution is:
```
x_s = (c_s/c_t) x_t + (a_s - (c_s/c_t) a_t) x_T + c_s ∫_{λ_t}^{λ_s} e^λ x_θ(x_{t_λ}, t_λ, x_T) dλ.
```
Taylor-expand `x_θ` in `λ` (with `h = λ_s - λ_t`); the scalar integrals are analytic:
```
∫ ≈ e^{λ_s}[ (1 - e^{-h}) x̂_t + (h - 1 + e^{-h}) x̂_t^{(1)} + (h^2/2 - h + 1 - e^{-h}) x̂_t^{(2)} ].
```
Estimate the `λ`-derivatives by finite differences over *past* predictions (no extra NFE): with
one previous time `u`, `x̂_t^{(1)} ≈ (x̂_t - x̂_u)/h_1`, `h_1 = λ_t - λ_u`; with two previous times,
```
x̂_t^{(1)} ≈ [ (x̂_t - x̂_{u1})/h1 (2h1 + h2) - (x̂_{u1} - x̂_{u2})/h2 · h1 ] / (h1 + h2),
x̂_t^{(2)} ≈ 2 [ (x̂_t - x̂_{u1})/h1 - (x̂_{u1} - x̂_{u2})/h2 ] / (h1 + h2).
```
First step (and optionally last) drops to first order.

## Working code

First-order, `η`-controlled DBIM sampler (the data predictor `denoiser(x, t) → x̂_0` is
NFE-counted; the schedule exposes `get_abc(t) → (a_t, b_t, c_t)` and
`get_alpha_rho(t) → (α_t, α_t/α_T, σ_t/α_t, …)`, where this schedule `rho` is not the injected
`ρ_n`; the loop names the current/larger time `s` and
the next/smaller target time `t`):

```python
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from .nn import append_dims
from .random_util import BatchedSeedGenerator


@torch.no_grad()
def sample_dbim(denoiser, diffusion, x, ts, eta=1.0, mask=None, seed=None, **kwargs):
    x_T = x
    path, pred_x0 = [], []
    ones = x.new_ones([x.shape[0]])
    indices = tqdm(range(len(ts) - 1), disable=(dist.get_rank() != 0))

    nfe = 0
    # first step: predict x_0 at the endpoint, seed the first interior bridge state with booting noise
    x0_hat = denoiser(x, diffusion.t_max * ones)
    generator = BatchedSeedGenerator(seed)
    noise = generator.randn_like(x0_hat)
    first_noise = noise                                   # booting noise = latent variable
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)   # a_t x_T + b_t x0_hat + c_t noise
    path.append(x.detach().cpu()); pred_x0.append(x0_hat.detach().cpu()); nfe += 1

    for _, i in enumerate(indices):
        s = ts[i]            # current (larger) time  = t_{n+1}
        t = ts[i + 1]        # next (smaller) time     = t_n

        x0_hat = denoiser(x, s * ones)
        if mask is not None:
            x0_hat = x0_hat * mask + x_T * (1 - mask)

        a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
        a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
        _, _, rho_s, _ = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_alpha_rho(s * ones)]
        alpha_t, _, rho_t, _ = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_alpha_rho(t * ones)]

        # per-step injected std rho_n = eta * sigma_{t_n} * sqrt(1 - SNR_{t_{n+1}}/SNR_{t_n})
        omega_st = eta * (alpha_t * rho_t) * (1 - rho_t**2 / rho_s**2).sqrt()

        tmp_var = (c_t**2 - omega_st**2).sqrt() / c_s     # recycled-direction coeff sqrt(c_t^2 - rho_n^2)/c_s
        coeff_xs = tmp_var
        coeff_x0_hat = b_t - tmp_var * b_s
        coeff_xT = a_t - tmp_var * a_s

        noise = generator.randn_like(x0_hat)
        # drop fresh noise on the final step (endpoint sharpness)
        x = (coeff_x0_hat * x0_hat + coeff_xT * x_T + coeff_xs * x
             + (1 if i != len(ts) - 2 else 0) * omega_st * noise)

        path.append(x.detach().cpu()); pred_x0.append(x0_hat.detach().cpu()); nfe += 1

    return x, path, nfe, pred_x0, ts, first_noise
```

Deterministic high-order variant (same harness slot; reuses past predictions to estimate the
`λ`-derivatives, first-order on the first and last steps). In this code, `s` is the current/larger
time and `t` is the next/smaller time, so `h = lambda_t - lambda_s` is the positive `λ` step:

```python
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from .nn import append_dims
from .random_util import BatchedSeedGenerator


@torch.no_grad()
def sample_dbim_high_order(denoiser, diffusion, x, ts, mask=None, order=2,
                           lower_order_final=True, seed=None, **kwargs):
    if order not in [2, 3]:
        raise NotImplementedError("Not supported")
    x_T = x
    path, pred_x0 = [], []
    ones = x.new_ones([x.shape[0]])
    indices = tqdm(range(len(ts) - 1), disable=(dist.get_rank() != 0))

    nfe = 0
    x0_hat = denoiser(x, diffusion.t_max * ones)
    generator = BatchedSeedGenerator(seed)
    noise = generator.randn_like(x0_hat)
    first_noise = noise
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)
    path.append(x.detach().cpu()); pred_x0.append(x0_hat.detach().cpu()); nfe += 1

    u = diffusion.t_max
    if u == 1.0:
        u -= 5e-5
    u = [u for _ in range(order - 1)]
    xu_hat = [x0_hat.detach().clone() for _ in range(order - 1)]

    for _, i in enumerate(indices):
        s = ts[i]; t = ts[i + 1]

        if (lower_order_final and i + 1 == len(ts) - 1) or (i == 0):     # first-order
            a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
            tmp_var = c_t / c_s
            coeff_xs, coeff_x0_hat, coeff_xT = tmp_var, b_t - tmp_var * b_s, a_t - tmp_var * a_s
            x0_hat = denoiser(x, s * ones); nfe += 1
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            x = coeff_xs * x + coeff_x0_hat * x0_hat + coeff_xT * x_T

        elif order == 2 or i == 1:                                       # second-order
            a_u, b_u, c_u = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(u[-1] * ones)]
            a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u, lambda_s, lambda_t = torch.log(b_u / c_u), torch.log(b_s / c_s), torch.log(b_t / c_t)
            x0_hat = denoiser(x, s * ones); nfe += 1
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            h, h2 = lambda_t - lambda_s, lambda_s - lambda_u
            integral = torch.exp(lambda_t) * ((1 - torch.exp(-h)) * x0_hat
                                              + (torch.exp(-h) + h - 1) * (x0_hat - xu_hat[-1]) / h2)
            x = x * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        elif order == 3:                                                 # third-order
            a_u1, b_u1, c_u1 = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(u[-1] * ones)]
            a_u2, b_u2, c_u2 = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(u[-2] * ones)]
            a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u2, lambda_u1, lambda_s, lambda_t = (torch.log(b_u2 / c_u2), torch.log(b_u1 / c_u1),
                                                        torch.log(b_s / c_s), torch.log(b_t / c_t))
            x0_hat = denoiser(x, s * ones); nfe += 1
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            h, h1, h2 = lambda_t - lambda_s, lambda_s - lambda_u1, lambda_u1 - lambda_u2
            dx0_hat = ((x0_hat - xu_hat[-1]) * (2 * h1 + h2) / h1 - (xu_hat[-1] - xu_hat[-2]) * h1 / h2) / (h1 + h2)
            d2x0_hat = 2 * ((x0_hat - xu_hat[-1]) / h1 - (xu_hat[-1] - xu_hat[-2]) / h2) / (h1 + h2)
            integral = torch.exp(lambda_t) * ((1 - torch.exp(-h)) * x0_hat
                                              + (torch.exp(-h) + h - 1) * dx0_hat
                                              + (h**2 / 2 - h + 1 - torch.exp(-h)) * d2x0_hat)
            x = x * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        u.append(s); u.pop(0)
        xu_hat.append(x0_hat); xu_hat.pop(0)
        path.append(x.detach().cpu()); pred_x0.append(x0_hat.detach().cpu())

    return x, path, nfe, pred_x0, ts, first_noise
```
