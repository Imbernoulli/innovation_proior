**Problem.** Sample the posterior `p(x | y) ∝ p(y | x) p(x)` for an inverse problem `y = A(x) + noise`,
using a pretrained diffusion model as the prior `p(x)`, training-free, for general (including nonlinear)
`A` and noisy `y`. The conditioning needs the time-level likelihood score `∇_{x_t} log p_t(y | x_t)`,
which has no closed form because `y` depends on `x_t` only through the unknown clean `x_0`.

**Key idea (the floor).** Marginalize `x_0`: `p(y | x_t) = E_{x_0 ∼ p(x_0|x_t)}[p(y | x_0)]`. Approximate
that expectation by the likelihood at the posterior *mean* — the Tweedie estimate `x_0_hat = E[x_0|x_t]`,
which the denoiser returns for free. Then the likelihood score is the gradient of
`−||y − A(x_0_hat)||²/(2 sigma_y²)`, taken **with respect to `x_t`** by backpropagating through the
denoiser (one VJP per step). No SVD, no projection: only `A` differentiable.

**Why it works / where it is weak.** Never enforcing exact consistency means noise is not burned in (good
for inpainting); needing only a differentiable `A` covers the nonlinear scattering operator. But the
point estimate collapses the denoising posterior to its mean, so it is least reliable at high noise where
the posterior is broad, and it costs one denoiser backward per step.

**Scaffold edit / hyperparameters.** Residual-normalize the squared-loss gradient with `0.5/sqrt(loss)`
so the step is a constant step on the residual *norm* (one stable knob `self.scale`). A per-`ENV`
`PROBLEM_CONFIGS` table sets `guidance_scale` per operator — `inv-scatter` 50.0 (clean large residual),
`blackhole` 1e-3 (sparse interferometer, different residual scale), `inpainting` 1.0 (small pixel-domain
loss; the default 50 explodes). `clip_grad` is opt-in and left **off** here, because clipping the raw
gradient before the `0.5/sqrt(loss)` rescaling would corrupt the tuned scales; `nan_to_num` is always on.
SDE reverse step; VP schedule, 1000 steps.

```python
import os
import torch
from tqdm import tqdm
from algo.base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler
import numpy as np


class Custom(Algo):
    """DPS: Diffusion Posterior Sampling.
    Score-based guidance using the gradient of the data likelihood.
    Requires forward_op.gradient() — best for differentiable forward operators.
    """

    # Per-problem optimized hyperparameters
    # inv-scatter: linear forward op, gradient clean → high guidance works
    # navier-stokes: PDE solver forward op, gradient VERY noisy/unstable →
    #   low guidance + gradient clipping to prevent NaN divergence
    # blackhole: non-trivial forward op → moderate guidance
    # clip_grad: ONLY enable for problems where the forward-op gradient is
    #   numerically unstable (e.g. NS PDE solver producing NaN). Clipping the
    #   raw ll_grad norm before the 1/sqrt(loss_scale) rescaling changes the
    #   effective guidance scale, so leave it OFF for well-behaved problems
    #   (inv-scatter / blackhole) to preserve their tuned guidance_scale.
    PROBLEM_CONFIGS = {
        'inv-scatter': {'guidance_scale': 50.0, 'clip_grad': False},
        'blackhole': {'guidance_scale': 1e-3, 'clip_grad': False},
        # FFHQ256 box-inpaint with sigma_noise=0.05. The default 50.0 (tuned for
        # the inv-scatter forward op) explodes here because the pixel-domain
        # data fitting loss is much smaller. guidance_scale=1.0 matches typical
        # DPS values for natural-image inverse problems.
        'inpainting': {'guidance_scale': 1.0, 'clip_grad': False},
    }

    def __init__(self, net, forward_op,
                 diffusion_scheduler_config=None,
                 guidance_scale=50.0,
                 sde=True,
                 **kwargs):
        super(Custom, self).__init__(net, forward_op)
        # Apply per-problem overrides
        env = os.environ.get('ENV', '')
        self.clip_grad = False
        if env in self.PROBLEM_CONFIGS:
            cfg = self.PROBLEM_CONFIGS[env]
            guidance_scale = cfg.get('guidance_scale', guidance_scale)
            self.clip_grad = cfg.get('clip_grad', False)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config or {
            'num_steps': 1000, 'schedule': 'vp', 'timestep': 'vp', 'scaling': 'vp'
        }
        # Override num_steps for expensive problems
        if env in self.PROBLEM_CONFIGS and 'num_steps' in self.PROBLEM_CONFIGS[env]:
            self.diffusion_scheduler_config['num_steps'] = self.PROBLEM_CONFIGS[env]['num_steps']
        self.scheduler = Scheduler(**self.diffusion_scheduler_config)
        self.sde = sde

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
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

            denoised = self.net(
                x_cur / self.scheduler.scaling_steps[i],
                torch.as_tensor(sigma).to(x_cur.device)
            )
            gradient, loss_scale = self.forward_op.gradient(
                denoised, observation, return_loss=True
            )

            ll_grad = torch.autograd.grad(denoised, x_cur, gradient)[0]
            # Clip gradient to prevent NaN (only needed for NS solver / acoustic);
            # for well-behaved problems (inv-scatter, blackhole) this would
            # corrupt the tuned guidance scale, so the clip is opt-in.
            if self.clip_grad:
                grad_norm = ll_grad.norm()
                max_grad_norm = 1.0
                if grad_norm > max_grad_norm:
                    ll_grad = ll_grad * (max_grad_norm / grad_norm)
            # Always replace NaN/Inf gradients with zero (cheap + safe)
            ll_grad = torch.nan_to_num(ll_grad, nan=0.0, posinf=0.0, neginf=0.0)
            ll_grad = ll_grad * 0.5 / torch.sqrt(loss_scale).clamp(min=1e-6)

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
