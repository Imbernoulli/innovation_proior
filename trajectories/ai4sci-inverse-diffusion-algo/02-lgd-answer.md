**Problem (from step 1).** DPS approximates the time-level likelihood `p_t(y | x_t)` by the likelihood at
the posterior *mean* `x_0_hat = E[x_0|x_t]`. That delta is structurally mis-scaled: too large at high
noise and too small at low noise, because evaluating a curved loss gradient at the mean is not the same
as averaging it over the (broad, at high noise) denoising posterior `p(x_0|x_t)`.

**Key idea.** Keep DPS's single-denoiser-call structure but replace the delta with a Gaussian surrogate
`q(x_0|x_t) = N(x_0_hat, r_t² I)`. The mean `x_0_hat` is the provably optimal fixed-covariance mean; the
width `r_t = sigma_t / sqrt(1 + sigma_t²)` is the conjugate posterior std — it widens to the prior at high
noise and collapses to the DPS delta at low noise (recovering DPS where DPS was right). Estimate
`E_{x_0∼q}[p(y|x_0)]` by Monte Carlo over `n` reparameterized samples `x^(i) = x_0_hat + r_t ε^(i)`,
which all branch off one denoiser output, so the network backward is paid **once** regardless of `n`. The
TV bound `2M·TV(p,q)` says a wider, posterior-matched `q` lowers the worst-case guidance bias.

**Why it works.** The bias is `2M·TV(p,q)`; a posterior-matched Gaussian is closer in TV than a point
mass (TV = 1), so the guidance is less biased exactly at the high-noise steps where DPS was worst.

**Scaffold edit / hyperparameters (what THIS task does).** The exact estimator differentiates a
log-mean-exp of the negative loss (softmin-weighted per-sample gradients). The harness instead takes the
**arithmetic mean** of the per-sample squared-residual gradients `g = (1/n) Σ ∇||A(x^(i)) − y||²`
(`batch_grad=True`) before one VJP through the denoiser — a least-squares mean-gradient approximation, not
the softmin weighting. Then root-loss-normalize with `0.5/sqrt(avg_loss)` (inherited from DPS) and
subtract from the EDM SDE step. Per-`ENV` table: `inv-scatter` guidance 3200.0 / `num_mc_samples` 20,
`blackhole` 1e-3 / 5, `inpainting` 1.0 / 5. VP schedule, 1000 steps.

```python
import os
import torch
from tqdm import tqdm
from algo.base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler
import numpy as np


class Custom(Algo):
    """LGD: Loss-Guided Diffusion.
    Uses Monte Carlo gradient estimation for measurement guidance.
    """

    # Per-problem task-local hyperparameters, initialized from InverseBench-style
    # inverse problem settings and then adjusted for this benchmark harness.
    # inpainting: ffhq256 box-mask, sigma_noise=0.05. Pixel-domain image prior
    #   has raw loss_scale ~1e3 at early steps so the 3200 default (tuned for
    #   the electromagnetic inv-scatter forward op) diverges immediately. Use
    #   guidance_scale=1.0 in the normalized image range, matching typical
    #   DPS-family values for natural-image inverse problems. num_mc_samples=5
    #   is a conservative middle ground — higher reduces variance but costs
    #   memory linearly in the samples dimension.
    PROBLEM_CONFIGS = {
        'inv-scatter': {'guidance_scale': 3200.0, 'num_mc_samples': 20},
        'navier-stokes': {'guidance_scale': 3e-3, 'num_mc_samples': 3},
        'blackhole': {'guidance_scale': 1e-3, 'num_mc_samples': 5},
        'acoustic': {'guidance_scale': 1.0, 'num_mc_samples': 3, 'num_steps': 100},
        'inpainting': {'guidance_scale': 1.0, 'num_mc_samples': 5},
    }

    def __init__(self, net, forward_op,
                 diffusion_scheduler_config=None,
                 guidance_scale=3200.0,
                 num_mc_samples=20,
                 batch_grad=True,
                 sde=True,
                 **kwargs):
        super(Custom, self).__init__(net, forward_op)
        # Apply per-problem overrides
        env = os.environ.get('ENV', '')
        if env in self.PROBLEM_CONFIGS:
            cfg = self.PROBLEM_CONFIGS[env]
            guidance_scale = cfg.get('guidance_scale', guidance_scale)
            num_mc_samples = cfg.get('num_mc_samples', num_mc_samples)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config or {
            'num_steps': 1000, 'schedule': 'vp', 'timestep': 'vp', 'scaling': 'vp'
        }
        # Override num_steps for expensive problems
        if env in self.PROBLEM_CONFIGS and 'num_steps' in self.PROBLEM_CONFIGS[env]:
            self.diffusion_scheduler_config['num_steps'] = self.PROBLEM_CONFIGS[env]['num_steps']
        self.scheduler = Scheduler(**self.diffusion_scheduler_config)
        self.sde = sde
        self.num_samples = num_mc_samples
        self.batch_grad = batch_grad

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        x_initial = torch.randn(
            num_samples, self.net.img_channels,
            self.net.img_resolution, self.net.img_resolution,
            device=device
        ) * self.scheduler.sigma_max
        x_next = x_initial
        x_next.requires_grad = True
        pbar = tqdm(range(self.scheduler.num_steps))

        for i in pbar:
            x_cur = x_next.detach().requires_grad_(True)

            sigma = self.scheduler.sigma_steps[i]
            factor = self.scheduler.factor_steps[i]
            scaling_factor = self.scheduler.scaling_factor[i]
            rt = sigma / np.sqrt(1 + sigma ** 2)

            denoised = self.net(
                x_cur / self.scheduler.scaling_steps[i],
                torch.as_tensor(sigma).to(x_cur.device)
            )

            samples = denoised + torch.randn(
                (self.num_samples, *denoised.shape[1:]), device=device
            ) * rt

            if self.batch_grad:
                gradient, loss_scale = self.forward_op.gradient(
                    samples, observation, return_loss=True
                )
                avg_loss = loss_scale
            else:
                gradients = torch.empty(
                    (self.num_samples, *denoised.shape[1:]), device=device
                )
                losses = np.empty(self.num_samples)
                for j in range(self.num_samples):
                    gradient, loss_scale = self.forward_op.gradient(
                        samples[j:j+1], observation, return_loss=True
                    )
                    gradients[j] = gradient
                    losses[j] = loss_scale
                avg_loss = losses.mean()
                gradient = gradients

            avg_grad = torch.mean(gradient, dim=0, keepdim=True).detach()

            ll_grad = torch.autograd.grad(denoised, x_cur, avg_grad)[0]
            ll_grad = ll_grad * 0.5 / torch.sqrt(avg_loss)

            score = (
                (denoised - x_cur / self.scheduler.scaling_steps[i])
                / sigma ** 2 / self.scheduler.scaling_steps[i]
            )
            pbar.set_description(
                f'Iteration {i + 1}/{self.scheduler.num_steps}. '
                f'Data fitting loss: {torch.sqrt(loss_scale)}'
            )

            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = (x_cur * scaling_factor + factor * score
                          + np.sqrt(factor) * epsilon)
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5
            x_next -= ll_grad * self.scale

        return x_next
```
