# Flow Matching

## Problem

Train a Continuous Normalizing Flow (CNF) — a generative model defined by a learned time-dependent vector field `v_t(x; θ)` whose ODE `dφ_t/dt = v_t(φ_t)`, `φ_0 = id`, pushes a Gaussian prior `p_0 = N(0, I)` to the data distribution `q` at `t = 1` — **without** the usual maximum-likelihood training that requires simulating the ODE inside every gradient step, and **without** restricting the transformation to a simple diffusion process.

## Key idea

Regress the network field directly onto a *target* vector field that generates a chosen probability path from noise to data. The ideal objective,

```
L_FM(θ) = E_{t~U[0,1], x~p_t(x)} || v_t(x) - u_t(x) ||²,
```

is intractable because the marginal field `u_t` generating the desired marginal path `p_t` has no closed form. Two results make it tractable:

1. **Mixture construction + marginal field.** Build `p_t` from per-example conditional paths, `p_t(x) = ∫ p_t(x|x_1) q(x_1) dx_1`, with `p_0(·|x_1) = N(0,I)` and `p_1(·|x_1)` concentrated at `x_1`. The field
   ```
   u_t(x) = ∫ u_t(x|x_1) [ p_t(x|x_1) q(x_1) / p_t(x) ] dx_1
   ```
   (the posterior average of conditional fields) generates `p_t` — verified by plugging into the continuity equation `∂_t p_t + div(p_t u_t) = 0`.

2. **Gradient equivalence (Conditional Flow Matching).** The tractable per-example objective
   ```
   L_CFM(θ) = E_{t~U[0,1], x_1~q(x_1), x~p_t(x|x_1)} || v_t(x) - u_t(x|x_1) ||²
   ```
   satisfies `L_FM(θ) = L_CFM(θ) + const` and therefore `∇_θ L_FM = ∇_θ L_CFM`. (Expand both squared norms; `||target||²` is θ-independent; the `||v||²` and cross terms have equal expectations under the two samplings because `p_t` unfolds into the mixture and the `1/p_t` in `u_t` cancels the measure in the cross term.)

So one only designs **conditional** paths and writes down their closed-form fields.

## Conditional Gaussian paths and their field

For `p_t(x|x_1) = N(μ_t(x_1), σ_t(x_1)² I)` with the canonical affine flow `ψ_t(x) = σ_t x + μ_t` (the simplest, swirl-free generator), the unique generating field is

```
u_t(x|x_1) = (σ_t'(x_1)/σ_t(x_1)) (x - μ_t(x_1)) + μ_t'(x_1),
```

with `'` denoting `d/dt`. The directly chosen paths can satisfy `μ_0=0, σ_0=1` and `μ_1=x_1, σ_1=σ_min`; the diffusion special cases below inherit their usual approximate/asymptotic noise endpoint.

Special cases by choice of `(μ_t, σ_t)`:
- **VE diffusion:** `μ_t=x_1, σ_t=σ_{1-t}` ⇒ `u_t = -(σ'_{1-t}/σ_{1-t})(x - x_1)`.
- **VP diffusion** (DDPM-style), `μ_t=α_{1-t}x_1, σ_t=√(1-α_{1-t}²)`, `α_t=e^{-T(t)/2}`, `T(t)=∫_0^t β`:
  ```
  u_t(x|x_1) = -(T'(1-t)/2) [ (e^{-T(1-t)}x - e^{-T(1-t)/2}x_1) / (1 - e^{-T(1-t)}) ].
  ```
  Coincides with the diffusion probability-flow ODE field — but obtained without any SDE/Fokker–Planck reasoning.
- **Optimal-Transport (linear) path:** `μ_t = t x_1, σ_t = 1 - (1-σ_min) t`, giving
  ```
  u_t(x|x_1) = (x_1 - (1-σ_min) x) / (1 - (1-σ_min) t),
  ```
  defined for all `t∈[0,1]`. Its conditional flow `ψ_t(x_0) = (1-(1-σ_min)t)x_0 + t x_1` is the OT displacement interpolant between the two endpoint Gaussians, so trajectories are straight lines at constant speed. The CFM target collapses to a constant per pair:
  ```
  L_CFM(θ) = E_{t,q(x_1),p(x_0)} || v_t(ψ_t(x_0)) - ( x_1 - (1-σ_min) x_0 ) ||².
  ```

The straight, constant-direction OT field is easier to regress and far cheaper to integrate at sampling (low NFE) than the curved, time-rotating diffusion field; it is also exact at the noise boundary, unlike diffusion paths which only approach pure noise asymptotically.

## Algorithm

**Training.** For each batch: `x_1 ~ q`, `x_0 ~ N(0,I)`, `t ~ U[0,1]`; form `x_t = ψ_t(x_0)`; minimize `|| v_θ(x_t, t) - u_t(x_t|x_1) ||²` (for the OT path, target `= x_1 - (1-σ_min)x_0`). Plain MSE, no `λ(t)` weighting, one network call, no ODE solve.

**Sampling.** Draw `x_0 ~ N(0,I)`, integrate `dx/dt = v_θ(x, t)` from `t=0` to `t=1` with an off-the-shelf ODE solver.

**Likelihood.** Integrate the instantaneous change of variables `d/dt log p_t(φ_t) = -div(v_t(φ_t))` alongside the flow (Hutchinson trace estimator for an unbiased divergence in high dimensions).

## Code

```python
import torch


def pad_t_like_x(t, x):
    """Reshape time vector t (bs,) to broadcast over x (bs, *dim)."""
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


def sample_prior_like(x):
    return torch.randn_like(x)


class TrainingPairBuilder:
    """OT Gaussian conditional path for vector-field regression.

    p_t(x|x1) = N(t*x1, (1 - (1 - sigma_min) t)^2 I)
    psi_t(x0) = (1 - (1 - sigma_min) t) x0 + t x1
    u_t(x_t|x1) = (x1 - (1 - sigma_min) x_t) / (1 - (1 - sigma_min) t)
                = x1 - (1 - sigma_min) x0           (constant in t along the straight path)
    """

    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    def sample_xt(self, x0, x1, t):
        t = pad_t_like_x(t, x1)
        return (1 - (1 - self.sigma_min) * t) * x0 + t * x1

    def compute_conditional_flow(self, x0, x1, t, xt):
        # Closed-form conditional vector field at x_t = psi_t(x0).
        # For xt from sample_xt this is x1 - (1 - sigma_min) x0, constant along the line.
        del x0
        t = pad_t_like_x(t, x1)
        return (x1 - (1 - self.sigma_min) * xt) / (1 - (1 - self.sigma_min) * t)

    def sample_location_and_conditional_flow(self, x0, x1, t=None):
        if t is None:
            t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        xt = self.sample_xt(x0, x1, t)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        return t, xt, ut


def train_step(pair_builder, v_theta, opt, x1):
    """One CFM gradient step: regress v_theta(x_t, t) onto the conditional field."""
    x0 = sample_prior_like(x1)
    t, xt, ut = pair_builder.sample_location_and_conditional_flow(x0, x1)
    loss = ((v_theta(xt, t) - ut) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def sample(v_theta, n, shape, steps=100, device="cpu"):
    """Integrate dx/dt = v_theta(x, t) from noise (t=0) to data (t=1)."""
    x = torch.randn(n, *shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * dt, device=device)
        x = x + v_theta(x, t) * dt  # Euler; any adaptive ODE solver (e.g. dopri5) works
    return x
```
