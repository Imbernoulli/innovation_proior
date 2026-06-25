# DAPS (Decoupled Annealing Posterior Sampling), distilled

DAPS turns a pretrained diffusion model into a posterior sampler for general — including nonlinear —
noisy inverse problems by *decoupling* consecutive sampling steps. Instead of bending a single reverse
step toward `y` (which couples the noisy-latent prior dynamics with the clean-variable likelihood and
makes nonlinear problems fragile), each annealing level (1) runs the *unconditional* probability-flow ODE
to turn the noisy `x_t` into a sharp clean estimate `x_0_hat`, (2) samples the clean conditional
`p(x_0 | x_t, y)` by Langevin dynamics *on the clean variable* — where the forward operator is evaluated
directly and no denoiser Jacobian appears — and (3) re-noises the corrected clean sample to the next lower
level. The time-marginals anneal to the true posterior as the noise drops.

## Problem it solves

Sample `x_0 ~ p(x_0 | y) ∝ p(y | x_0) p(x_0)` for `y = A(x_0) + n`, `n ~ N(0, sigma_y^2 I)`, with `p(x_0)`
a pretrained diffusion prior, training-free, stable on *nonlinear* `A` (phase retrieval, nonlinear
deblurring, HDR) where trajectory-guidance solvers (DPS, LGD, ΠGDM) are most brittle.

## Key idea

1. **Decouple the steps.** The prior's score lives on the noisy `x_t`; the likelihood lives on the clean
   `x_0`. Single-step guidance forces both onto `x_t`, coupling them so a bad guidance step propagates
   down the trajectory. DAPS separates them and stitches with re-noising.

2. **Prior update (unconditional).** Solve the unconditional PF-ODE from `x_t` down to clean to get
   `x_0_hat`, a sharp sample of `p(x_0 | x_t)` — sharper than DPS's single Tweedie call, affordable
   because the steps are decoupled (a few Euler NFE).

3. **Likelihood correction (clean-variable Langevin).** Sample `p(x_0 | x_t, y) ∝ N(x_0_hat, r_t^2 I)
   p(y | x_0)`, `r_t ≈ sigma`, whose score is `−(x_0 − x_0_hat)/r_t^2 − ∇_{x_0}||A(x_0) − y||^2/(2
   sigma_y^2)`. Run `N` Langevin steps

   ```
   x_0 <- x_0 - eta (x_0 - x_0hat)/r_t^2 - eta grad||A(x_0)-y||^2/(2 sigma_y^2) + sqrt(2 eta) z.
   ```

   `A` is evaluated at `x_0` directly — **no denoiser Jacobian**, so nonlinear `A` is no harder than
   linear. The denoiser is outside this loop, so `N = 100` is cheap.

4. **Re-noise.** `x_{t-1} ~ N(x_0^{(N)}, sigma_next^2 I)`. Each iterate is freshly built from a clean
   correction, not inherited as a small step — that is the decoupling, and why errors do not propagate.

5. **Anneals to the posterior.** Wide/data-dominated conditional at high noise, sharpening to
   `p(x_0 | y)` as `sigma → 0`; the Langevin noise keeps it a *sampler* (not RED-diff's mode-seeking
   collapse).

## Defaults and why

- **Annealing schedule:** EDM rho-schedule, `sigma_max ≈ 80` to `sigma_min`, ~`N_A` levels (the
  standard EDM discretization).
- **ODE sub-sampler:** Euler PF-ODE from `sigma`, a few NFE per level (configurable 4–44).
- **Langevin:** `N = 100` steps (50 for LatentDAPS); step size `eta` decayed over the inner loop so the
  `1/sigma^2` anchor stays `O(1)`; `sigma_y = 0.05` is the known measurement noise.

## Working code (canonical reference form)

Faithful to the official DAPS sampler: an outer annealing loop, an unconditional PF-ODE prior update, a
clean-variable Langevin inner loop, and a re-noising step.

```python
import torch
import numpy as np
from .base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler


class LangevinDynamics:
    """Sample p(x0 | x_t, y) on the CLEAN variable. The denoiser is NOT here —
    only the cheap forward operator, so many steps are affordable."""

    def __init__(self, num_steps, lr, sigma_y, lr_min_ratio=0.01):
        self.num_steps = num_steps
        self.lr = lr
        self.sigma_y = sigma_y
        self.lr_min_ratio = lr_min_ratio

    def sample(self, x0hat, forward_op, observation, sigma):
        x = x0hat.clone().detach()
        for j in range(self.num_steps):
            ratio = j / self.num_steps
            lr = self.lr * (self.lr_min_ratio + (1 - self.lr_min_ratio)
                            * 0.5 * (1 + np.cos(np.pi * ratio)))
            x = x.detach().requires_grad_(True)
            data_grad, _ = forward_op.gradient(x, observation, return_loss=True)
            grad = (x - x0hat) / (sigma ** 2) + data_grad / (2 * self.sigma_y ** 2)
            x = x.detach() - lr * grad + np.sqrt(2 * lr) * torch.randn_like(x)
        return x.detach()


class DAPS(Algo):
    def __init__(self, net, forward_op, annealing_scheduler_config,
                 diffusion_scheduler_config, lgvd_lr, lgvd_steps=100, sigma_y=0.05):
        super().__init__(net, forward_op)
        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.lgvd = LangevinDynamics(lgvd_steps, lgvd_lr, sigma_y)

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
        sigmas = self.annealing_scheduler.sigma_steps
        x_t = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution,
                          self.net.img_resolution, device=device) * sigmas[0]

        for i in range(self.annealing_scheduler.num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < len(sigmas) else 0.0

            # (1) unconditional PF-ODE prior update: x_t -> x0hat
            diff_sched = Scheduler(sigma_max=sigma, **self.diffusion_scheduler_config)
            x0hat = DiffusionSampler(diff_sched).sample(self.net, x_t, SDE=False)

            # (2) clean-variable Langevin correction: sample p(x0 | x_t, y)
            x0y = self.lgvd.sample(x0hat, self.forward_op, observation, sigma)

            # (3) re-noise to the next lower level
            x_t = x0y + torch.randn_like(x0y) * sigma_next

        return x_t
```
