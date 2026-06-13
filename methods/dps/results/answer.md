# Diffusion Posterior Sampling (DPS), distilled

DPS turns a pre-trained unconditional diffusion model into a posterior sampler for a general
(possibly nonlinear) noisy inverse problem `y = A(x_0) + n`, without retraining. At every
reverse-diffusion step it adds a soft **likelihood-gradient guidance** term, computed by
backpropagating a data-fidelity loss through the in-loop Tweedie estimate of the clean signal.
It needs only the unconditional score and a differentiable forward operator — no SVD, no
projection onto a measurement subspace.

## Problem it solves

Sample from `p(x_0 | y) ∝ p(y | x_0) p(x_0)` with `p(x_0)` given by a pre-trained diffusion
prior, for a known forward operator `A` (linear or nonlinear) and measurement noise `n`
(Gaussian or Poisson), correctly in the noisy regime and for general `A`.

## Key idea

By Bayes' rule the conditional score splits:

```
grad_{x_t} log p_t(x_t | y) = grad_{x_t} log p_t(x_t) + grad_{x_t} log p_t(y | x_t).
```

The first term is the pre-trained score `s_theta(x_t, t)`. The second, the **likelihood score
at noise level `t`**, is intractable because `y` depends on `x_t` only through the unknown
clean `x_0`. Make it tractable in two moves:

1. **Marginalize `x_0` (exact).** Using the conditional independence `y ⊥ x_t | x_0`,

   ```
   p(y | x_t) = ∫ p(y | x_0) p(x_0 | x_t) dx_0 = E_{x_0 ~ p(x_0|x_t)} [ p(y | x_0) ].
   ```

2. **Point-estimate at the posterior mean (approximate).** `p(x_0 | x_t)` is intractable, but
   its mean is closed-form via **Tweedie's formula**:

   ```
   x_0_hat := E[x_0 | x_t] = (1/sqrt(alpha_bar(t))) ( x_t + (1 - alpha_bar(t)) s_theta(x_t, t) ).
   ```

   Replace the expectation of the likelihood with the likelihood at the mean:

   ```
   p(y | x_t) ≈ p(y | x_0_hat).
   ```

The error is a Jensen gap. For a `q`-dimensional Gaussian measurement density
`h_sigma(u) = N(y; u, sigma^2 I)` viewed as a normalized density in its mean argument,
`||grad_u h_sigma(u)||` is maximized at `||u-y|| = sigma`, giving the Euclidean Lipschitz
constant
`L_sigma = e^{-1/2} (2 pi)^(-q/2) sigma^(-(q+1))`, so

```
J  <=  L_sigma || grad_x A(x) || m_1,
m_1 = ∫ ||x_0 - x_0_hat|| p(x_0 | x_t) dx_0,
```

with `|| grad_x A(x) ||` the operator's max Jacobian norm. Norm variants can add dimension
factors, but for the normalized density the sigma dependence remains `sigma^(-(q+1))`: the
bound is loose at very small `sigma` and tends to `0` as `sigma -> infinity`, so it **improves
with more measurement noise** - exactly the regime where exact-consistency projection solvers
fail.

## Guidance term and update

Differentiating the surrogate likelihood (gradient taken w.r.t. `x_t`, i.e. backprop through
`x_0_hat(x_t)` and through `A`):

```
Gaussian:  grad_{x_t} log p_t(y | x_t) ≈ - rho * grad_{x_t} || y - A(x_0_hat) ||_2^2,
           rho = 1/(2 sigma^2) for an unhalved squared norm, or 1/sigma^2 for a half-squared loss.
Poisson :  grad_{x_t} log p_t(y | x_t) ≈ - rho * grad_{x_t} || y - A(x_0_hat) ||_Lambda^2,  [Lambda]_jj = 1/(2 y_j)
```

The direct Poisson surrogate has factors `(y_j / [A(x_0_hat)]_j - 1)` and is unstable near zero
predicted intensity. The Poisson guidance uses the Gaussian/shot-noise approximation of Poisson
(`mu ≈ N(mu, mu)`, variance frozen to `y_j`), giving the fixed-weight quadratic above. Constants
such as the Gaussian likelihood coefficient and the Poisson global scale are absorbed into the
discrete guidance scale in implementations.
Because the gradient is just a backprop through the differentiable `A`, **nonlinear operators
need no special handling** (Fourier phase retrieval, neural forward models, or other
differentiable measurement maps).

Plugged into the ancestral / EDM reverse step (unconditional update `x'_{i-1}`, then guidance):

```
x_0_hat   = (1/sqrt(alpha_bar_i)) ( x_i + (1 - alpha_bar_i) s_theta(x_i, i) )
x'_{i-1}  = (sqrt(alpha_i)(1 - alpha_bar_{i-1})/(1 - alpha_bar_i)) x_i
            + (sqrt(alpha_bar_{i-1}) beta_i/(1 - alpha_bar_i)) x_0_hat + sigma_tilde_i z,   z ~ N(0, I)
x_{i-1}   = x'_{i-1} - zeta_i grad_{x_i} || y - A(x_0_hat) ||_2^2
```

## Step size

The literal noise-variance coefficient is fixed while the squared-residual gradient magnitude
swings by orders of magnitude across the trajectory. Use **residual normalization**:

```
zeta_i = zeta' / || y - A(x_0_hat(x_i)) ||,    zeta' constant.
```

Since `grad ||r||^2 = 2 ||r|| grad ||r||`, scaling by `zeta_i = zeta'/||r||` makes the update a
constant step on the **un-squared** norm `||y - A(x_0_hat)||`, up to a factor of `2` that is
absorbed into the step-size constant. InverseBench implements the exact root-loss gradient by
multiplying the squared-loss VJP by `0.5/sqrt(loss)`.

## Relation to prior methods

- **MCG (Chung et al. 2022a)**: the DPS Gaussian guidance term (with `W = I`) is MCG's
  manifold-constrained gradient — the projection of the data-fidelity force onto the tangent
  space `T_{x_0_hat} M` of the data manifold. MCG additionally **projects onto the measurement
  subspace** `{A x = y}` after the gradient step. In the noiseless exact-consistency setting,
  adding that projection recovers the full MCG sampler; in noisy problems DPS omits it so
  corrupted measurements are not enforced exactly.
- **Projection / POCS (score-SDE, ILVR)**: replaced by a soft likelihood gradient; no exact
  consistency, so no noise amplification, and nonlinear `A` is supported.
- **Spectral (DDRM/SNIPS)**: no SVD needed.
- **Linear-likelihood Langevin (robust-CSGM)**: uses the correct time-level surrogate
  `p(y | x_0_hat)` instead of the `t=0` term `A^H(y - Ax)/sigma^2` patched by an annealing
  heuristic, and is not restricted to linear `A`.

## Working code

Filling the conditioning slot of the EDM reverse-diffusion harness. The gradient w.r.t. `x_t`
is a vector-Jacobian product through `x_0_hat`, and the residual normalization is folded into
the `0.5/sqrt(loss)` factor, turning `grad ||.||^2` into `grad ||.||`; `self.scale` is the
remaining guidance constant.

```python
import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler
import numpy as np


class DPS(Algo):
    """Diffusion Posterior Sampling. Unconditional reverse step + soft likelihood
    guidance via the gradient of the data-fidelity loss through the Tweedie
    estimate. Requires a differentiable forward operator (handles nonlinear A)."""

    def __init__(self, net, forward_op, diffusion_scheduler_config, guidance_scale, sde=True):
        super().__init__(net, forward_op)
        self.scale = guidance_scale                      # zeta': single step-size budget
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sde = sde

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
        x_initial = torch.randn(num_samples, self.net.img_channels,
                                self.net.img_resolution, self.net.img_resolution,
                                device=device) * self.scheduler.sigma_max
        x_next = x_initial
        x_next.requires_grad = True

        pbar = tqdm(range(self.scheduler.num_steps))
        for i in pbar:
            x_cur = x_next.detach().requires_grad_(True)     # graph for the VJP through x_0_hat

            sigma = self.scheduler.sigma_steps[i]
            factor = self.scheduler.factor_steps[i]
            scaling_factor = self.scheduler.scaling_factor[i]

            # Tweedie posterior mean x_0_hat = E[x_0 | x_t]
            denoised = self.net(x_cur / self.scheduler.scaling_steps[i],
                                torch.as_tensor(sigma).to(x_cur.device))

            # grad_{x_0_hat} ||A(x_0_hat) - y||^2 and loss = ||A(x_0_hat) - y||^2
            gradient, loss_scale = self.forward_op.gradient(denoised, observation, return_loss=True)

            # VJP: backprop through x_0_hat(x_cur) -> grad_{x_cur} ||y - A(x_0_hat)||^2
            ll_grad = torch.autograd.grad(denoised, x_cur, gradient)[0]
            # residual-normalize: 0.5/sqrt(loss) turns grad||.||^2 into grad||.||
            ll_grad = ll_grad * 0.5 / torch.sqrt(loss_scale)

            # unconditional reverse step (EDM-scaled score; SDE or PF-ODE)
            score = ((denoised - x_cur / self.scheduler.scaling_steps[i])
                     / sigma ** 2 / self.scheduler.scaling_steps[i])
            pbar.set_description(f'Iteration {i + 1}/{self.scheduler.num_steps}. '
                                 f'Data fitting loss: {torch.sqrt(loss_scale)}')

            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5

            # likelihood guidance: subtract zeta' * (residual-normalized fidelity gradient)
            x_next = x_next - ll_grad * self.scale
        return x_next
```
