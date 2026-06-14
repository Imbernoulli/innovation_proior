# DPM-Solver++(2S), distilled

DPM-Solver++(2S) is a training-free, second-order **singlestep** solver for the diffusion ODE,
written on the **data-prediction** (clean-image, `x0`) parameterization. It is the second-order
member of a family whose first-order member is DDIM. Each step makes two denoiser calls: one at the
start, one at an intermediate point in half-log-SNR; the finite difference of the two clean-image
predictions estimates the first derivative, giving second-order accuracy. Writing the solver on the
data prediction (rather than the noise prediction) does two things at once: it exposes a clean-image
estimate at every step, so thresholding composes directly, and it puts a strictly-smaller-than-one
factor on the high-order error term, which keeps the high-order step stable under the large guidance
scales that destabilize noise-prediction high-order solvers.

## Problem it solves

Fast (≈15-20 NFE) training-free sampling of guided diffusion models at *large* guidance scale,
where (1) high-order noise-prediction solvers become unstable — worse than first-order DDIM, and
worse as order grows — because guidance amplifies the model's `lambda`-derivatives and shrinks the
solver's convergence radius, and (2) the converged clean image falls out of the data range
(train-test mismatch), needing a clean-image estimate to clip.

## Setup

Forward process `x_t = alpha_t x_0 + sigma_t eps`, `eps ~ N(0, I)`, SNR `alpha_t^2/sigma_t^2`
strictly decreasing. Two equivalent denoiser views: noise `eps_theta(x_t, t)` and data
`x_theta(x_t, t) = (x_t - sigma_t eps_theta)/alpha_t` (Tweedie clean-data estimate). Half-log-SNR
`lambda_t := log(alpha_t/sigma_t)`, strictly decreasing in `t`, inverse `t_lambda`; one step uses
`h := lambda_t - lambda_s > 0`.

## Key idea

**Solve the data-prediction ODE.** Its exact solution from `s` to `t` (proved by differentiating
back to the ODE) is

```
x_t = (sigma_t/sigma_s) x_s + sigma_t * integral_{lambda_s}^{lambda_t} e^{lambda} x_hat_theta(x_hat_lambda, lambda) dlambda.
```

This makes `sigma_t/sigma_s` exact and integrates `e^{+lambda} x_theta` — a different discretization
target from the noise-prediction solution (which makes `alpha_t/alpha_s` exact and integrates
`e^{-lambda} eps_theta`).

**Taylor + analytic integrals.** Expand `x_hat_theta(lambda)` around `lambda_s` to order `k-1` and
integrate each term against `e^{lambda}` by parts:

```
integral e^{lambda} dlambda                 -> sigma_t * (...) = -alpha_t(e^{-h} - 1) = alpha_t(1 - e^{-h})
integral e^{lambda}(lambda - lambda_s) dlambda -> sigma_t * (...) = alpha_t(h - 1 + e^{-h})
```

- **k = 1 (DDIM, x0 form):** `x_t = (sigma_t/sigma_s) x_s - alpha_t(e^{-h} - 1) x_theta(x_s, s)`.
- **k = 2 (the 2S step, singlestep):** intermediate point at `lambda_{s1} = lambda_s + r1 h`
  (default `r1 = 1/2`), reached by a first-order sub-step
  `u = (sigma_{s1}/sigma_s) x_s - alpha_{s1}(e^{-r1 h} - 1) x_theta(x_s, s)`; estimate
  `x_theta^{(1)}(lambda_s) ≈ (x_theta(u, s1) - x_theta(x_s, s))/(r1 h)`. Using the standard
  stiff-order simplification `(h - 1 + e^{-h})/h ≈ -(e^{-h} - 1)/2 = h/2 + O(h^2)`, the update is a
  weighted combination of the two clean-image predictions:

```
D = (1 - 1/(2 r1)) x_theta(x_s, s) + (1/(2 r1)) x_theta(u, s1)
x_t = (sigma_t/sigma_s) x_s - alpha_t(e^{-h} - 1) D
```

  With the canonical `r1 = 1/2`, `D = x_theta(u, s1)` (midpoint), matching
  `solver_type="dpmsolver"`.

## Why it works (the two payoffs of the data parameterization)

1. **Smaller error constant under guidance.** Rewriting the 2S update into noise-prediction terms
   makes it *identical* to the noise-prediction second-order solver except for an extra factor
   `e^{-r1 h} < 1` (since `h > 0`) multiplying the first-derivative correction term. Since
   `eps(intermediate) - eps(start) = r1 h eps^{(1)}(start) + O(h^2)`, this is the term where
   derivative-estimation error enters. Smaller constant on that term ⇒ smaller
   discretization error ⇒ stable even when guidance amplifies the derivatives.
2. **Thresholding for free.** `x_theta` is in hand every call, so a clipped data model
   `x_hat_theta = clip(x_theta)` (elementwise / dynamic thresholding) fixes the train-test mismatch
   for pixel-space data. For latent-space models (Stable Diffusion / SDXL) thresholding is off
   (latent unbounded), but payoff (1) still holds.

## Convergence

Second order: `||x_{t_M} - x_tilde_{t_M}|| = O(h_max^2)`, assuming `x_theta` total derivatives to
order 2 exist and are continuous, `x_theta` is `L`-Lipschitz in `x`, `h_max = O(1/M)`, and
`r_i > c > 0`. The local truncation error of the second-order term is `O(h^3)` (the leading constant
is `h^3/12` from `(h - 1 + e^{-h}) + (e^{-h} - 1)h/2 = h^3/12 + O(h^4)`); over `O(1/h)` steps this
gives global `O(h^2)`. Propagation: `Delta_i <= (alpha_{t_i}/alpha_{t_{i-1}}) Delta_{i-1} +
C h_i(Delta_{i-1} + h_i^2)` ⇒ `Delta_i = O(h_max^2)` (the homogeneous factor is
`alpha_{t_i}/alpha_{t_{i-1}}`, carried through the `x_theta`-Lipschitz error bound, not the
update's `sigma_{t_i}/sigma_{t_{i-1}}` linear coefficient).

## Budget and schedule

Singlestep: 2 NFE/step, so an `N`-NFE budget yields about `N/2` second-order boundaries; an order
scheduler may reserve one first-order (DDIM) boundary for odd budgets or very small step counts. Use
`solver_order = 2` for guided sampling (higher order needs even smaller steps under guidance).
Uniform-`t` step spacing for high-resolution guided sampling. Use `expm1` for the `(e^{...} - 1)`
factors (numerical stability for small `h`).

## Working code

Filling the per-step `update_rule` of the sampler harness, in the data-prediction (`x0`) convention.
Mirrors the canonical singlestep second-order update (`algorithm_type="dpmsolver++"`,
`solver_type="dpmsolver"`, `r1=0.5`): the model wrapper returns the (optionally thresholded) data
prediction, and the solver sees a clean `x0` oracle.

```python
import torch


class NoiseSchedule:
    """Provides alpha(t), sigma(t)=sqrt(1-alpha^2), lambda(t)=log(alpha/sigma), inverse_lambda."""
    def alpha(self, t): ...
    def sigma(self, t): ...
    def lamb(self, t): return torch.log(self.alpha(t)) - torch.log(self.sigma(t))
    def inverse_lamb(self, lamb): ...


class DPMSolverPP2S:
    """DPM-Solver++(2S): second-order singlestep solver on the data-prediction diffusion ODE."""

    def __init__(self, model, noise_schedule, cfg_guidance=7.5, r1=0.5, threshold=None):
        self.model = model                    # frozen denoiser -> (eps_uncond, eps_cond)
        self.ns = noise_schedule
        self.cfg_guidance = cfg_guidance      # guidance scale s
        self.r1 = r1                          # intermediate fraction in lambda (default midpoint)
        self.threshold = threshold            # x0 clip (pixel-space); None for latent models

    def data_prediction(self, x, t, uc, c):
        """x0 = (x - sigma_t * eps_tilde) / alpha_t  (Tweedie), optionally thresholded."""
        eps_uncond, eps_cond = self.model(x, t, uc, c)
        eps = eps_uncond + self.cfg_guidance * (eps_cond - eps_uncond)   # guided noise
        alpha_t, sigma_t = self.ns.alpha(t), self.ns.sigma(t)
        x0 = (x - sigma_t * eps) / alpha_t
        if self.threshold is not None:
            x0 = self.threshold(x0)
        return x0

    def first_order_update(self, x, s, t, uc, c, model_s=None):
        """DPM-Solver-1 == DDIM (data-prediction form), from s to t."""
        h = self.ns.lamb(t) - self.ns.lamb(s)
        phi_1 = torch.expm1(-h)                                    # e^{-h} - 1
        if model_s is None:
            model_s = self.data_prediction(x, s, uc, c)
        return (self.ns.sigma(t) / self.ns.sigma(s)) * x - (self.ns.alpha(t) * phi_1) * model_s

    def second_order_update(self, x, s, t, uc, c):
        """DPM-Solver++(2S): one singlestep second-order update from s to t."""
        ns, r1 = self.ns, self.r1
        lambda_s, lambda_t = ns.lamb(s), ns.lamb(t)
        h = lambda_t - lambda_s                                    # > 0 (lambda increases as t->0)
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lamb(lambda_s1)                            # intermediate time

        phi_11 = torch.expm1(-r1 * h)                             # e^{-r1 h} - 1
        phi_1 = torch.expm1(-h)                                   # e^{-h} - 1

        model_s = self.data_prediction(x, s, uc, c)              # x0 at start (1st network call)
        x_s1 = (ns.sigma(s1) / ns.sigma(s)) * x - (ns.alpha(s1) * phi_11) * model_s   # DDIM to s1
        model_s1 = self.data_prediction(x_s1, s1, uc, c)         # x0 at s1 (2nd network call)

        # x_t = (sigma_t/sigma_s) x - alpha_t(e^{-h}-1)[ model_s + (1/(2 r1))(model_s1 - model_s) ]
        return (
            (ns.sigma(t) / ns.sigma(s)) * x
            - (ns.alpha(t) * phi_1) * model_s
            - (0.5 / r1) * (ns.alpha(t) * phi_1) * (model_s1 - model_s)
        )

    @torch.no_grad()
    def sample(self, x_T, timesteps, uc, c, orders=None, lower_order_final=False):
        """timesteps are decreasing step boundaries; order-2 boundaries cost two network calls."""
        x = x_T
        n = len(timesteps) - 1
        if orders is None:
            orders = [2] * n
        for i in range(n):
            s, t = timesteps[i], timesteps[i + 1]
            use_first_order = orders[i] == 1 or (lower_order_final and i == n - 1)
            if use_first_order:
                x = self.first_order_update(x, s, t, uc, c)       # DDIM final step
            else:
                x = self.second_order_update(x, s, t, uc, c)
        return x
```

## Relation to prior methods

- **DDIM** (Song et al. 2021) = the **first-order** member of this family (its `eta = 0`,
  data-prediction form is exactly the `k = 1` update). DPM-Solver++(2S) is the second-order
  generalization of deterministic DDIM with respect to the data prediction.
- **DPM-Solver-2** (Lu et al. 2022) = the analogous singlestep second-order solver on the *noise*
  prediction. DPM-Solver++(2S) differs by exactly one factor `e^{-r1 h} < 1` on the high-order error
  term ⇒ smaller error constant ⇒ more stable under large guidance.
- **DPM-Solver++(2M)** (the multistep sibling) reuses the previous step's data prediction instead of
  an intermediate point: `D = (1 + 1/(2 r)) x_theta(t_{i-1}) - (1/(2 r)) x_theta(t_{i-2})` with
  `r = h_{i-1}/h_i`, then `x_t = (sigma_t/sigma_s) x_s - alpha_t(e^{-h} - 1) D`. One new call per
  step (N steps for N NFE, smaller `h`), at the cost of depending on history; 2S keeps the cleaner
  history-independent local error constant at 2 NFE/step.
- **Thresholding** (Saharia et al. 2022) composes directly because `x_theta` is exposed; pixel-space
  only.
