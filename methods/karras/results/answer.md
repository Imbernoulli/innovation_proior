The schedule places the sequence of noise levels a diffusion ODE sampler steps through by warping a
uniform grid through a power law. Given a step budget `N` and noise-level endpoints, it
interpolates *linearly in the warped coordinate* `sigma^{1/rho}` and then undoes the warp:

```
sigma_{i<N} = ( sigma_max^{1/rho} + (i/(N-1)) ( sigma_min^{1/rho} - sigma_max^{1/rho} ) )^rho,
sigma_N     = 0,
```

for `i = 0, ..., N-1`, giving `N+1` nodes `sigma_0 = sigma_max > sigma_1 > ... > sigma_{N-1} = sigma_min > sigma_N = 0`.
Defaults: `rho = 7`, `sigma_min = 0.002`, `sigma_max = 80` (for image data of scale `sigma_data = 0.5`).

## Problem it solves

Diffusion sampling integrates an ODE whose right-hand side is a learned denoiser; the cost is dominated
by the number of network evaluations (NFE). With a small step budget the numerical truncation error
dominates output quality. The discretization `{sigma_i}` decides how the few steps — and hence the
per-step error — are distributed across the noise range. The goal is to reduce the damaging part of that
error at a given `N` (so `N` can be pushed down), decoupled from how the denoiser was trained so the same
recipe transfers across models.

## Key idea

1. **Decouple the schedule from training and from theoretical convenience.** The sampler sees the
   denoiser as a black box; `{sigma_i}`, `sigma(t)`, `s(t)` are free dials to be chosen for sampling
   accuracy, not inherited from the forward noising process.
2. **Straighten the trajectory first.** With `sigma(t) = t`, `s(t) = 1` the ODE is
   `dx/dt = (x - D(x; t)) / t`; a single Euler step to `t = 0` lands on the denoiser output, so the
   tangent always points at the current denoised estimate, which changes slowly with noise level —
   trajectories are nearly straight except in a narrow middle band, minimizing curvature and thus error.
3. **Shrink steps where error lives.** Measured single-step truncation error is large at low `sigma`,
   small at high `sigma`, and nearly independent of the sample (so one deterministic schedule suffices).
   The step size must decrease as `sigma` decreases.
4. **A one-parameter warp family.** Place `sigma_{i<N} = w(A i + B)` with a monotone unbounded warp `w`
   and endpoints fixing `A, B`. The power law `w(z) = z^rho` gives the closed form above. It contains the
   prior schedules as limits: `rho = 1` is uniform-in-`sigma`; `rho -> infinity` is the geometric
   (log-uniform / variance-exploding) sequence. `rho` is the single knob for how much resolution goes to
   low noise.
5. **Pick `rho` for perception, not for numerical balance.** With Heun's second-order solver the error is
   nearly equalized across `sigma` at `rho ≈ 3` (optimal for raw RGB-space accuracy, e.g. ODE inversion).
   But generation is judged by FID, and accuracy at high `sigma` is perceptually cheap (`sigma_max` is
   nearly arbitrary) while accuracy at low `sigma` is decisive. So push past the balance point:
   **`rho = 7`** trades tolerable high-noise error for reduced low-noise error and is robustly good across
   models and step counts.

## Solver pairing

Heun's method (two-stage second-order Runge–Kutta, `a = 1`): Euler predictor, re-evaluate at the
endpoint, average the two slopes — local error `O(h^3)` vs Euler's `O(h^2)` at one extra evaluation per
step. `a = 1` is near-optimal experimentally and, uniquely among RK2 variants, lands the extra evaluation
exactly on a schedule node `t_{i+1}` (so models trained at discrete noise levels stay usable). Fall back
to a plain Euler step on the final hop to `sigma = 0` (the corrector divides by `sigma`).

## Defaults and why

- `rho = 7`: perceptual-FID sweet spot, deliberately above the `rho ≈ 3` that merely balances numerical
  error, because high-noise accuracy is perceptually cheap and low-noise accuracy is decisive.
- `sigma_min = 0.002`: lowest *resolved* noise level; below it discerning the noise is hard and irrelevant.
  The final Euler step takes `sigma_min -> 0`.
- `sigma_max = 80`: large compared to the data scale (`sigma_data ≈ 0.5`), so the start is effectively pure
  Gaussian noise; its exact value barely affects the output distribution — which is also why coarse steps
  there are cheap.
- `N+1` nodes from `N` steps; strictly decreasing by construction (`z_i` linear-decreasing, `z -> z^rho`
  increasing for `z > 0`).

## Working code

The schedule (terminating at clean data, `sigma_N = 0`):

```python
import torch


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Noise-level schedule: power-law warp of a uniform grid.

    n solver steps -> n+1 nodes (sigma_0 = sigma_max ... sigma_{n-1} = sigma_min, sigma_n = 0).
    """
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)                       # sigma_min^{1/rho}
    max_inv_rho = sigma_max ** (1 / rho)                       # sigma_max^{1/rho}
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)                      # append terminal sigma_n = 0
```

The second-order Heun sampler it feeds (with `sigma(t) = t`, `s(t) = 1`, Euler fallback at zero):

```python
@torch.no_grad()
def sample_heun(denoiser, x, sigmas):
    """x: initial noise scaled to sigma_max (= randn(...) * sigmas[0])."""
    for i in range(len(sigmas) - 1):
        sigma_cur, sigma_next = sigmas[i], sigmas[i + 1]
        d = (x - denoiser(x, sigma_cur)) / sigma_cur                 # dx/dt at sigma_cur
        x_pred = x + (sigma_next - sigma_cur) * d                    # Euler predictor
        if sigma_next == 0:
            x = x_pred                                               # final hop: Euler only
        else:
            d_prime = (x_pred - denoiser(x_pred, sigma_next)) / sigma_next
            x = x + (sigma_next - sigma_cur) * (0.5 * d + 0.5 * d_prime)   # trapezoidal correction
    return x
```

The canonical PyTorch sampler form folds the same schedule and
gamma-based stochastic churn into one loop:

```python
import numpy as np
import torch


@torch.no_grad()
def edm_sampler(net, latents, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
                S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, randn_like=torch.randn_like):
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time-step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho)
               + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        # Optional churn (stochastic sampling); S_churn=0 gives deterministic Heun.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 2nd-order correction (skip on the last step into sigma=0).
        if i < num_steps - 1:
            denoised = net(x_next, t_next).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next
```

## Relation to prior schedules

- **Uniform in `sigma`** = this schedule with `rho = 1`.
- **Geometric / log-uniform (variance-exploding) schedule** = this schedule with `rho -> infinity`; the
  power-law family is a parametric generalization that interpolates between uniform and geometric.
- **Cosine / variance-preserving schedules** are training-time forward-process discretizations reused for
  sampling; this schedule is instead derived from per-step sampling integration error, then set to the
  FID-favorable side of the numerical-error balance point.
