**Problem (from step 2).** DPS and LGD both bend the *same coupled* reverse step toward `y` — the prior
score (on the noisy latent) and the likelihood correction (on the clean signal) entangled in one update,
each paying a denoiser Jacobian. That is fragile: LGD's sampled surrogate gained on `inv-scatter` /
`inpainting` PSNR but blew up `blackhole`'s chi-squared (cp_chi2 5.89 → 17.58).

**Key idea.** Stop bending the trajectory. Pose inference on the clean `x_0` directly: fit a Gaussian
`q(x_0|y) = N(mu, sigma² I)` to the posterior by minimizing `KL(q || p(x_0|y))`. Bayes-expand to a
reconstruction term `||y − f(mu)||²` plus `KL(q || p(x_0))`; the ML-diffusion identity turns the prior-KL
into a time-integrated score-matching loss. With `sigma → 0` and integration by parts (requiring
`omega(0) = 0` so the boundary dies), the regularizer gradient collapses to the **denoising residual**
`lambda_t (epsilon_theta(x_t;t) − epsilon)` — one frozen-denoiser forward pass, **no Jacobian**. Sampling
becomes Adam optimization of `mu`. The signal-domain identity `mu − mu_hat_t = (sigma_t/alpha_t)(eps_theta
− eps)` forces `lambda_t = lambda/SNR_t`.

**Why it works.** The decoupled objective separates fitting `mu` to `y` from regularizing it by the prior,
so the per-operator balance is a data weight on a *stable* optimization, not a guidance scale on a touchy
coupled step — which is what should stop blackhole's chi-squared blow-up while fitting inv-scatter harder.

**Scaffold edit / hyperparameters (what THIS task does).** Maintain `mu`; each step noise it to `xt`, one
detached denoiser call for `pred_epsilon`, gradient `= observation_weight · ∇_mu ||f(mu) − y||² + lam ·
(pred_epsilon − epsilon)` assigned to `mu.grad`, Adam step (`betas=(0.9, 0.99)`, no weight decay). The
data gradient is computed *directly on `mu`* (no denoiser backward). `lambda_scheduling_type='constant'`
default; the scheduler `sigma` is `1/SNR`, so `'linear'` would give `lambda/SNR_t`. Per-`ENV`:
`inv-scatter` obs_weight 1500 / lr 0.04 / lambda 5e-4; `blackhole` obs_weight 1e-4 / lr 1e-2 / lambda
0.25; `inpainting` omitted (defaults 1500 / 0.04 / 5e-4 already work). VP schedule, 1000 steps.

```python
import os
import torch
import tqdm
from algo.base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler
import numpy as np


class Custom(Algo):
    """REDDiff: Regularization by Denoising with Diffusion priors.
    Optimization-based approach using diffusion score as regularizer.
    """

    # Per-problem task-local hyperparameters, initialized from InverseBench-style
    # inverse problem settings and then adjusted for this benchmark harness.
    # 'inpainting' is intentionally omitted: the default __init__ values
    # (observation_weight=1500, base_lr=0.04, base_lambda=5e-4) already work
    # well on FFHQ256 box-inpaint (REDDiff achieves PSNR~22 with these), so
    # adding an override here would only risk regressing the result.
    PROBLEM_CONFIGS = {
        'inv-scatter': {'observation_weight': 1500.0, 'base_lr': 0.04, 'base_lambda': 5e-4},
        'blackhole': {'observation_weight': 1e-4, 'base_lr': 1e-2, 'base_lambda': 0.25},
    }

    def __init__(self, net, forward_op,
                 num_steps=1000,
                 observation_weight=1500.0,
                 base_lambda=5e-4,
                 base_lr=0.04,
                 lambda_scheduling_type='constant',
                 **kwargs):
        super(Custom, self).__init__(net, forward_op)
        # Apply per-problem overrides
        env = os.environ.get('ENV', '')
        if env in self.PROBLEM_CONFIGS:
            cfg = self.PROBLEM_CONFIGS[env]
            observation_weight = cfg.get('observation_weight', observation_weight)
            base_lr = cfg.get('base_lr', base_lr)
            base_lambda = cfg.get('base_lambda', base_lambda)
            num_steps = cfg.get('num_steps', num_steps)
        self.net.eval().requires_grad_(False)

        self.scheduler = Scheduler(
            num_steps=num_steps, schedule='vp',
            timestep='vp', scaling='vp'
        )
        self.base_lr = base_lr
        self.observation_weight = observation_weight
        if lambda_scheduling_type == 'linear':
            self.lambda_fn = lambda sigma: sigma * base_lambda
        elif lambda_scheduling_type == 'sqrt':
            self.lambda_fn = lambda sigma: torch.sqrt(sigma) * base_lambda
        elif lambda_scheduling_type == 'constant':
            self.lambda_fn = lambda sigma: base_lambda
        else:
            raise NotImplementedError

    def pred_epsilon(self, model, x, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        d = model(x, sigma)
        return (x - d) / sigma

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        num_steps = self.scheduler.num_steps
        pbar = tqdm.trange(num_steps)
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)

        mu = torch.zeros(
            num_samples, self.net.img_channels,
            self.net.img_resolution, self.net.img_resolution,
            device=device
        ).requires_grad_(True)
        optimizer = torch.optim.Adam([mu], lr=self.base_lr, betas=(0.9, 0.99))

        for step in pbar:
            with torch.no_grad():
                sigma = self.scheduler.sigma_steps[step]
                scaling = self.scheduler.scaling_steps[step]
                epsilon = torch.randn_like(mu)
                xt = scaling * (mu + sigma * epsilon)
                pred_epsilon = self.pred_epsilon(self.net, xt, sigma).detach()

            lam = self.lambda_fn(sigma)
            optimizer.zero_grad()

            gradient, loss_scale = self.forward_op.gradient(
                mu, observation, return_loss=True
            )
            gradient = (gradient * self.observation_weight
                        + lam * (pred_epsilon - epsilon))
            mu.grad = gradient

            optimizer.step()
            pbar.set_description(
                f'Iteration {step + 1}/{num_steps}. '
                f'Data fitting loss: {torch.sqrt(loss_scale)}'
            )
        return mu
```
