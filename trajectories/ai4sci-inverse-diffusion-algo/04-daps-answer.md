**Problem (from step 3).** RED-diff fixed the coupling that broke LGD (blackhole `cp_chi2` 17.58 → 3.01)
by decoupling the likelihood from the trajectory and optimizing the clean `mu`. But it commits to a single
Gaussian mode (`sigma → 0`), so on the multimodal sparse-interferometer posterior it lands a
*consistent-but-blurred* image: blackhole PSNR stuck at 21.86 despite excellent chi-squared. It stopped
being a *sampler*.

**Key idea.** Keep RED-diff's decoupling but restore stochastic exploration on the clean variable, and
re-inject the prior separately. Per annealing level `sigma`: (1) **prior update** — unconditional PF-ODE
from `x_t` to a sharp clean estimate `x0hat` (sharper than one Tweedie call); (2) **likelihood
correction** — Langevin-sample `p(x0 | x_t, y) ∝ N(x0hat, sigma² I) p(y | x0)`, score
`−(x0 − x0hat)/sigma² − ∇_{x0}||A(x0) − y||²·data_scale`, with injected noise `sqrt(2·lr)·z` — `A`
evaluated *at x0 directly*, no denoiser Jacobian; (3) **re-noise** `x_t ← x0^{(N)} + sigma_next·z`. The
re-noising decouples consecutive iterates so errors do not propagate; the Langevin noise keeps it a true
sampler that anneals to `p(x0 | y)`.

**Why it should clear the bar.** The Langevin noise is exactly what RED-diff discarded, so it should lift
the under-explored blackhole PSNR above 21.86; the decoupling + re-noising are what let it do so *without*
re-breaking the chi-squared (the operators no longer share a fragile propagating latent); clean-variable
`A` keeps the nonlinear inv-scatter operator stable.

**Scaffold edit / hyperparameters.** `DiffusionSampler(Scheduler(sigma_max=sigma, ...)).sample(net, x_t,
SDE=False)` is the unconditional ODE prior update (canonical DAPS builds the inner scheduler with
`sigma_max=sigma`). Langevin: `N=100` steps, step size cosine-decayed so the `1/sigma²` anchor stays
`O(1)`. Per-`ENV` `data_scale` and `lr` (the same per-operator residual-scale adaptation DPS/LGD/REDDiff
all need): `inv-scatter` large `data_scale` (clean EM residual), `blackhole` tiny `data_scale` (sparse
residual), `inpainting` moderate. `sigma_y=0.05`. Annealing: VP schedule.

```python
import os
import torch
from tqdm import tqdm
from algo.base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler
import numpy as np


class Custom(Algo):
    """DAPS: Decoupled Annealing Posterior Sampling.
    Each annealing level: (1) unconditional PF-ODE prior update x_t -> x0hat;
    (2) clean-variable Langevin correction sampling p(x0 | x_t, y); (3) re-noise.
    The denoiser is outside the Langevin loop, so A is evaluated on the clean
    variable with no network Jacobian (robust for nonlinear operators).
    """

    # Per-problem hyperparameters. The three forward operators produce residuals
    # on very different scales, so the data-term weight and Langevin step size are
    # set per ENV (the same per-operator adaptation DPS/LGD/REDDiff all need);
    # the algorithm itself is unchanged across problems.
    #   inv-scatter: clean large EM residual  -> large data_scale.
    #   blackhole:   sparse interferometer    -> tiny data_scale.
    #   inpainting:  small pixel-domain loss (sigma_y=0.05) -> moderate data_scale.
    PROBLEM_CONFIGS = {
        'inv-scatter': {'data_scale': 1.0, 'lgvd_lr': 1e-3, 'num_annealing_steps': 200},
        'blackhole': {'data_scale': 1e-5, 'lgvd_lr': 1e-3, 'num_annealing_steps': 200},
        'inpainting': {'data_scale': 1.0 / (2 * 0.05 ** 2), 'lgvd_lr': 5e-4,
                       'num_annealing_steps': 200},
    }

    def __init__(self, net, forward_op,
                 diffusion_scheduler_config=None,
                 num_annealing_steps=200,
                 lgvd_lr=1e-3,
                 lgvd_steps=100,
                 data_scale=1.0,
                 sigma_y=0.05,
                 lr_min_ratio=0.01,
                 ode_steps=5,
                 **kwargs):
        super(Custom, self).__init__(net, forward_op)
        env = os.environ.get('ENV', '')
        if env in self.PROBLEM_CONFIGS:
            cfg = self.PROBLEM_CONFIGS[env]
            data_scale = cfg.get('data_scale', data_scale)
            lgvd_lr = cfg.get('lgvd_lr', lgvd_lr)
            num_annealing_steps = cfg.get('num_annealing_steps', num_annealing_steps)
        self.data_scale = data_scale
        self.lgvd_lr = lgvd_lr
        self.lgvd_steps = lgvd_steps
        self.sigma_y = sigma_y
        self.lr_min_ratio = lr_min_ratio
        self.ode_steps = ode_steps
        # Outer annealing schedule over noise levels (VP).
        self.annealing_scheduler = Scheduler(
            num_steps=num_annealing_steps, schedule='vp',
            timestep='vp', scaling='vp'
        )
        # Config for the inner unconditional PF-ODE sub-sampler (sigma_max set per level).
        self.diffusion_scheduler_config = diffusion_scheduler_config or {
            'num_steps': self.ode_steps, 'schedule': 'vp',
            'timestep': 'vp', 'scaling': 'vp'
        }

    def langevin_sample(self, x0hat, observation, sigma):
        """Sample p(x0 | x_t, y) on the clean variable by Langevin dynamics.
        Score = -(x0 - x0hat)/sigma^2 - data_scale * grad||A(x0) - y||^2."""
        x = x0hat.clone().detach()
        sigma2 = float(sigma) ** 2 + 1e-8
        for j in range(self.lgvd_steps):
            ratio = j / self.lgvd_steps
            lr = self.lgvd_lr * (self.lr_min_ratio + (1 - self.lr_min_ratio)
                                 * 0.5 * (1 + np.cos(np.pi * ratio)))
            x = x.detach().requires_grad_(True)
            data_grad, _ = self.forward_op.gradient(x, observation, return_loss=True)
            data_grad = torch.nan_to_num(data_grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad = (x.detach() - x0hat) / sigma2 + self.data_scale * data_grad
            x = x.detach() - lr * grad + np.sqrt(2 * lr) * torch.randn_like(x)
        return x.detach()

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
        sigmas = self.annealing_scheduler.sigma_steps
        num_steps = self.annealing_scheduler.num_steps

        x_t = torch.randn(
            num_samples, self.net.img_channels,
            self.net.img_resolution, self.net.img_resolution,
            device=device
        ) * self.annealing_scheduler.sigma_max

        pbar = tqdm(range(num_steps))
        for i in pbar:
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < len(sigmas) else 0.0

            # (1) PRIOR UPDATE: unconditional PF-ODE from x_t (inner sigma_max = sigma).
            diff_sched = Scheduler(sigma_max=float(sigma), **self.diffusion_scheduler_config)
            with torch.no_grad():
                x0hat = DiffusionSampler(diff_sched).sample(self.net, x_t, SDE=False)
            x0hat = x0hat.detach()

            # (2) LIKELIHOOD CORRECTION: Langevin sample of p(x0 | x_t, y) on the clean var.
            x0y = self.langevin_sample(x0hat, observation, sigma)

            # (3) RE-NOISE to the next, lower annealing level.
            x_t = x0y + torch.randn_like(x0y) * float(sigma_next)

            loss = self.forward_op.loss(x0y, observation).mean()
            pbar.set_description(
                f'Annealing {i + 1}/{num_steps}. '
                f'Data fitting loss: {torch.sqrt(loss.clamp(min=0))}'
            )
        return x_t
```
