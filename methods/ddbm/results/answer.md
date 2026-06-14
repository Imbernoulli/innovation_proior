# Denoising Diffusion Bridge Models (DDBM)

## Problem

Diffusion models transport between data and *Gaussian noise*; image-to-image translation,
restoration, and inpainting need transport between *two arbitrary, paired* distributions
`(x_0, x_T) ~ p_data(x, y)` — clean target and observed source. DDBM is a principled generative
process between two such endpoints that (a) trains with a closed-form Gaussian marginal and a
closed-form objective, (b) reuses diffusion's preconditioning, noise schedules, and high-order
samplers, and (c) contains plain diffusion and flow matching as special cases.

## Key idea

Use Doob's h-transform to condition a base diffusion to hit a fixed endpoint, pin both ends to get a
*diffusion bridge*, define the training marginal as the doubly-pinned Gaussian, learn the bridge score
by denoising score matching, and generate by integrating the bridge's reverse-time SDE / probability-flow
ODE — with a hybrid stochastic-Euler + deterministic-Heun sampler that injects noise so a data-pinned
process produces sharp, diverse targets instead of a blurry conditional mean.

## The bridge

For a base diffusion with Gaussian kernel `p(x_t|x_0) = N(α_t x_0, σ_t² I)` and `SNR_t = α_t²/σ_t²`,
the doubly-pinned marginal (Bayes: `q(x_t|x_0,x_T) = p(x_T|x_t)p(x_t|x_0)/p(x_T|x_0)`) is Gaussian:

```
q(x_t | x_0, x_T) = N( mu_hat_t , sigma_hat_t^2 I )
  mu_hat_t      = (SNR_T/SNR_t)(alpha_t/alpha_T) x_T  +  alpha_t (1 - SNR_T/SNR_t) x_0
  sigma_hat_t^2 = sigma_t^2 (1 - SNR_T/SNR_t)
```

The mean linearly interpolates the (scaled) endpoints; the variance is 0 at both ends (Dirac on `x_T` at
`t=T`, on `x_0` at `t=0`). Renaming `x_t = a_t x_T + b_t x_0 + sqrt(v_t) eps`:
`a_t=(α_t/α_T)(SNR_T/SNR_t)`, `b_t=α_t(1-SNR_T/SNR_t)`, `v_t=σ_t²(1-SNR_T/SNR_t)`.

## Reverse processes (Theorem)

The conditional `q(x_t|x_T)` evolves by a Fokker-Planck with drift `f + g² h` (Kolmogorov forward for
`p(x_t|x_0)`, backward for `p(x_T|x_t)`, recombined by the product rule). Its time reversal:

```
reverse SDE:  dx_t = [ f(x_t,t) - g^2(t) ( s(x_t,t,y,T) - h(x_t,t,y,T) ) ] dt + g(t) dw_bar
PF-ODE:       dx_t = [ f(x_t,t) - g^2(t) ( (1/2) s(x_t,t,y,T) - h(x_t,t,y,T) ) ] dt
```

with `s = ∇log q(x_t|x_T)` (learned) and `h = ∇log p(x_T|x_t)` (Doob h-transform, closed form).
**Only the score `s` is halved in the ODE; `h` stays at full strength** (the ½ comes from the
continuity-equation conversion acting on `q`'s score, not on the bridge's defining h-drift).

## Training: denoising bridge score matching

```
L(theta) = E_{(x0,xT)~p_data, x_t~q(x_t|x0,xT), t}[ w(t) || s_theta(x_t, x_T, t)
                                                     - grad_xt log q(x_t|x0,xT) ||^2 ]
```

The `L₂` minimizer equals `∇log q(x_t|x_T)` (the conditioning on `x_T` rides along). The target
conditional score `-(x_t - mu_hat_t)/sigma_hat_t²` is closed form.

## pred-x0 parameterization (generalized EDM)

`D_θ(x_t,t) = c_skip x_t + c_out F_θ(c_in x_t; c_noise)` predicts `x_0`; the bridge score is
`s ≈ -(x_t - (a_t x_T + b_t D_θ))/v_t`. The canonical code stores `c_t=sqrt(v_t)` and therefore divides
by `c_t**2`. With endpoint stats `σ_0², σ_T², σ_{0T}` (variances and
covariance), require unit variance of input and of the effective target and minimize `c_out` w.r.t.
`c_skip`:

```
c_in(t)    = 1 / sqrt( a_t^2 sigma_T^2 + b_t^2 sigma_0^2 + 2 a_t b_t sigma_0T + v_t )
c_skip(t)  = ( b_t sigma_0^2 + a_t sigma_0T ) * c_in(t)^2
c_out(t)   = sqrt( a_t^2 (sigma_0^2 sigma_T^2 - sigma_0T^2) + sigma_0^2 v_t ) * c_in(t)
w(t)       = 1 / c_out(t)^2
c_noise(t) = (1/4) log t
```

The only new endpoint statistics vs EDM are `σ_T, σ_{0T}`. With `x_T = x_0 + Tε` (`σ_T²=σ_0²+T²`,
`σ_{0T}=σ_0²`) these reduce exactly to EDM: `c_in=1/√(σ_0²+t²)`, `c_skip=σ_0²/(σ_0²+t²)`,
`c_out=σ_0 t/√(σ_0²+t²)`. A guidance strength can scale the `h` term in the ODE (classifier-guidance
analogy); the canonical sampler functions below keep `h` at full strength.

## Special cases

- **Diffusion** (`p_data(x_0,x_T)=p(x_T|x_0)p_data(x_0)`, `x_T` Gaussian): marginalizing `x_T` makes the
  h-drift integrate to 0, recovering the diffusion reverse SDE/ODE and marginal `N(α_t x_0, σ_t² I)`.
- **OT-Flow-Matching / Rectified Flow**: VE bridge `σ_t²=c²t`, variance scaled by `c`, `c→0` gives ODE
  drift `→ (x_T - x_0)/T`, the straight-line velocity; match the drift since the bridge score diverges.

## Hybrid sampler

A pure ODE from the fixed data start `x_T` yields the blurry conditional mean, so inject noise
predictor-corrector style: on the decreasing EDM `ρ=7` grid, take a **stochastic Euler "churn"** step
(reverse SDE, `κ=1`) from `ts[i]` to `t_hat = ts[i] + r(ts[i+1]-ts[i])`, followed by a **deterministic
Heun** step (PF-ODE, `κ=½`) from `t_hat` to `ts[i+1]`; step ratio `r` trades exploration vs. endpoint
sharpness (`r≈1/3` for translation, `r=0` for generation). On the final step to `t=0`, use a single Euler
step.

```python
import torch
from tqdm.auto import tqdm
import torch.distributed as dist

from .nn import append_dims

def get_d(denoiser, noise_schedule, x, x_T, t, stochastic):
    ones = x.new_ones([x.shape[0]])
    f_t, g2_t = [append_dims(item, x.ndim) for item in noise_schedule.get_f_g2(t * ones)]
    alpha_t, alpha_bar_t, _, rho_bar_t = [
        append_dims(item, x.ndim) for item in noise_schedule.get_alpha_rho(t * ones)
    ]
    a_t, b_t, c_t = [append_dims(item, x.ndim) for item in noise_schedule.get_abc(t * ones)]
    denoised = denoiser(x, t * ones)
    grad_logq = -(x - (a_t * x_T + b_t * denoised)) / c_t**2
    grad_logpxTlxt = -(x - alpha_bar_t * x_T) / (alpha_t**2 * rho_bar_t**2)
    d = f_t * x - g2_t * ((0.5 if not stochastic else 1) * grad_logq - grad_logpxTlxt)
    return d, g2_t, denoised


def ddbm_simulate(denoiser, noise_schedule, x, x_T, t_cur, t_next, stochastic, second_order=False):
    dt = t_next - t_cur
    if isinstance(noise_schedule, I2SBNoiseSchedule):
        dt = dt * (noise_schedule.n_timestep - 1)
    d, g2_t, pred_x0 = get_d(denoiser, noise_schedule, x, x_T, t_cur, stochastic)
    x_new = x + d * dt + (0 if not stochastic else 1) * torch.randn_like(x) * (dt.abs() ** 0.5) * g2_t.sqrt()
    if second_order:
        d_2, _, pred_x0 = get_d(denoiser, noise_schedule, x_new, x_T, t_next, stochastic)
        d_prime = (d + d_2) / 2
        x_new = x + d_prime * dt + (0 if not stochastic else 1) * torch.randn_like(x) * (dt.abs() ** 0.5) * g2_t.sqrt()
    return x_new, pred_x0


@torch.no_grad()
def sample_heun(denoiser, diffusion, x, ts, churn_step_ratio=0.0, **kwargs):
    x_T = x
    path, pred_x0, nfe = [], [], 0
    indices = tqdm(range(len(ts) - 1), disable=(dist.get_rank() != 0))
    assert churn_step_ratio < 1

    for _, i in enumerate(indices):
        if churn_step_ratio > 0:
            t_hat = (ts[i + 1] - ts[i]) * churn_step_ratio + ts[i]
            x, _p = ddbm_simulate(denoiser, diffusion.noise_schedule, x, x_T, ts[i], t_hat, stochastic=True)
            nfe += 1
            path.append(x.detach().cpu()); pred_x0.append(_p.detach().cpu())
        else:
            t_hat = ts[i]
        if ts[i + 1] == 0:
            x, _p = ddbm_simulate(denoiser, diffusion.noise_schedule, x, x_T, t_hat, ts[i + 1], stochastic=False)
            nfe += 1
        else:
            x, _p = ddbm_simulate(denoiser, diffusion.noise_schedule, x, x_T, t_hat, ts[i + 1],
                                  stochastic=False, second_order=True)
            nfe += 2
        path.append(x.detach().cpu()); pred_x0.append(_p.detach().cpu())
    return x, path, nfe, pred_x0, ts, None
```

The EDM `ρ=7` time grid is `t_i = (t_max^{1/ρ} + i/(N-1)(t_min^{1/ρ} - t_max^{1/ρ}))^ρ` with a trailing
0. Each Heun iteration costs 3 denoiser calls (1 churn-Euler + 2 Heun); the final Euler-only iteration
costs 2 (churn + Euler). At a high-NFE reference (`ρ=7`, `N=17`, churn ratio `⅓`): 16 Heun + 1 final =
50 calls.
