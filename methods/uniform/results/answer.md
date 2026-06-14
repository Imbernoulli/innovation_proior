# Uniform (linear) sampling-time schedule

The uniform schedule places the few sampling times of a diffusion / diffusion-bridge sampler at
**constant spacing in the sampler's native time variable `t`**, from the noisy endpoint `t_max`
down to the data endpoint `t_min`. It is the "no-information" baseline placement: with the model
and the sampler update rule frozen, the only freedom is *where* to evaluate the denoiser, and
even spacing is the placement that assumes nothing about where the sampling trajectory curves.
In the discrete index-based setting the same idea is a **constant stride** over the trained time
indices, `tau_i = floor(c * i)` (the "linear" timestep selection), versus the curvature-warped
`tau_i = floor(c * i^2)` ("quadratic") sibling.

## Problem it solves

A trained denoiser is sampled by an integrator that marches a fixed update rule along `t` from
`t_max` to `t_min`, calling the denoiser once per step; the cost is the number of calls (NFE).
Under a tiny budget (`n = 5`), with the model and update rule frozen, the schedule of evaluation
times — an ordered list `t_max = t_0 > t_1 > ... > t_n = t_min` — is the only lever, and it must
strictly decrease, end exactly at `t_min`, have length `n + 1`, and carry no dataset-specific
constants (it must generalize across workloads).

## Key idea

Sampling the deterministic member of the model's generative family is Euler integration of its
probability-flow ODE; the schedule is the integrator's node placement. The Euler local error on
an interval of length `Δt_i` scales like `(Δt_i)^2` times the local trajectory curvature `C_i`,
so the total error is `~ Σ_i (Δt_i)^2 C_i`.

- **If the curvature were known**, minimizing `Σ_i (Δt_i)^2 C_i` subject to `Σ_i Δt_i = t_max -
  t_min` gives the Lagrange condition `2 C_i Δt_i = λ`, hence `Δt_i ∝ 1 / C_i` — shorten steps
  where the trajectory is hard. The warped schedules (EDM power-law `t^{1/ρ}`, log-linear,
  cosine) are exactly such bets, each baking in a fixed assumption about where curvature lives
  plus a shape hyperparameter.
- **With no curvature information** (the baseline regime, and the only honest stance for a
  cross-workload schedule), the problem is minimax: choose the grid before seeing `C_i`, against
  an adversary who can place a fixed curvature budget `Σ_i C_i = B` anywhere. Any non-uniform grid
  has a largest interval `Δt_max > (t_max - t_min)/n`; the adversary dumps the budget there and
  the error is `~ (Δt_max)^2 B`. The grid with no oversized interval to exploit — all `Δt_i`
  equal — minimizes the worst case. That is the **uniform grid**.

Equal spacing is therefore both the minimax-optimal grid under no curvature knowledge and the
direct inheritance of the dense sampler's even integer spacing. It carries **no shape
hyperparameter** (only the count `n`), so it transfers across workloads unchanged, and it hits both
endpoints exactly, so the terminal node lands on `t_min` with no correction.

## Why space evenly in `t` (not a warped coordinate)

The warped schedules are *also* uniform — uniform in `t^{1/ρ}` or in `log t`. Spacing evenly in a
transformed coordinate is a warp of the grid in `t`, which presupposes the trajectory is smoothest
in that coordinate — a curvature assumption. The frozen sampler's update rule is written in `t`
(it indexes its coefficients by `t` and queries `eps_theta(x_t, t)`), so `t` is the integrator's
native variable; constant `Δt` presupposes nothing about the warp.

## Final form

Continuous time (the bridge sampler marches a continuous `t`):

```
t_i = t_max + (i / n) (t_min - t_max),   i = 0, 1, ..., n
    -> t_0 = t_max,  t_n = t_min,  constant step Δt = (t_max - t_min)/n
```

Discrete index space (constant stride over the `T` trained indices):

```
tau_i = floor(c * i)      # "linear" / uniform selection
tau_i = floor(c * i^2)    # "quadratic" warped sibling, packing indices toward one end
```

Both are constant-step placements; the continuous form is the limit where the step need not be an
integer.

## Working code

The continuous schedule, filling the time-placement slot of the frozen bridge sampler:

```python
import torch


def get_sampling_times(n, t_min, t_max, device="cpu"):
    """Uniform (linear) time schedule for a frozen few-step bridge sampler.

    Places the n + 1 evaluation times at constant spacing in t, from t_max down to
    t_min. The no-information baseline: minimax-optimal node placement when the
    trajectory's curvature could sit anywhere, carrying no shape hyperparameter so it
    transfers across workloads unchanged.

    Contract:
      1. Length:     1-D tensor of length n + 1.
      2. Monotonic:  strictly decreasing (t_max > t_min => constant negative step).
      3. Terminal:   final element == t_min exactly (linspace hits both endpoints).
      4. Device:     returned on the requested device.
    For this task n = 5 (NFE = 5).
    """
    # t_i = t_max + (i/n)(t_min - t_max); linspace gives exact endpoints, no terminal fix.
    return torch.linspace(t_max, t_min, n + 1).to(device)
```

The discrete index-space twin (constant stride; the form used to select a sampling-timestep
subsequence from a model trained over `T` integer indices):

```python
import numpy as np


def make_uniform_integer_timesteps(num_sampling_timesteps, num_training_timesteps):
    """Uniform / linear stride over integer training indices.

    Mathematically this is tau_i = floor(c * i); the quadratic sibling is
    tau_i = floor(c * i**2). The common implementation realizes the linear rule
    with range(0, T, c), then shifts by +1 for alpha indexing.
    """
    c = num_training_timesteps // num_sampling_timesteps
    selected_timesteps = np.asarray(list(range(0, num_training_timesteps, c)))
    return selected_timesteps + 1
```

## What it claims (and does not)

It does not claim to be the best schedule on any given workload; a warp tuned to that workload's
true curvature would beat it there. It claims to be the disciplined "no bet" — minimax-optimal
without curvature information, hyperparameter-free, cross-workload by construction, and exact at
both endpoints — the baseline every informed schedule must justify itself against.
