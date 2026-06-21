## Research question

A pretrained diffusion model supplies a denoiser `net(x_t, sigma)` that returns the Tweedie posterior mean `E[x_0 | x_t]` at noise level `sigma`. The inverse problem is: given a known forward operator `A` and an observation `y = A(x) + noise`, reconstruct `x` by conditioning the diffusion prior on the measurement.

The object of study is the **measurement-conditioning rule**: how the denoiser, the noise schedule, the forward operator, and its gradient are combined into a sampler that draws from `p(x | y) ∝ p(y | x) p(x)`. The denoiser weights, forward operators, and evaluation tasks are fixed; only the conditioning rule is designed.

## Prior art / Background / Baselines

- **Reverse-SDE / score-based generative modeling.** Generation follows the reverse of the diffusion SDE using the score `∇_{x_t} log p_t(x_t)` supplied by the denoiser. Conditioning on `y` splits the score into the prior score plus the likelihood score `∇_{x_t} log p_t(y | x_t)`.
- **Projection-onto-measurement-subspace solvers (score-SDE, ILVR).** Take an unconditional diffusion step, then project the iterate onto `{x : A x = y}` to enforce data consistency.
- **Spectral-domain solvers (SNIPS / DDRM).** Diagonalize `A` by its SVD and treat measurement-domain Gaussian noise in closed form in the spectral basis.
- **Tweedie's formula.** The denoiser's output is a closed-form estimate of the posterior mean `E[x_0 | x_t]` at any noise level.

## Fixed substrate / Code framework

A pretrained EDM-style diffusion stack is frozen:

- `net(x, sigma)` → Tweedie estimate `E[x_0 | x_t]`; `net.img_channels`, `net.img_resolution` give the signal shape.
- `forward_op.forward(x)` → `A(x)`; `forward_op.gradient(x, y, return_loss=True)` → `(∇_x ||A(x) - y||², ||A(x) - y||²)`; `forward_op.loss(x, y)` → `||A(x) - y||²` per batch element; `forward_op.device`.
- `Scheduler(...)` → noise schedule exposing `.sigma_steps`, `.factor_steps`, `.scaling_factor`, `.scaling_steps`, `.num_steps`, `.sigma_max`.
- `DiffusionSampler(scheduler).sample(model, x_start, SDE=False)` → unconditional (prior-only) sampling.

The pretrained weights, forward-operator definitions, and evaluation problems are fixed.

## Editable interface

Only the `Custom` class in `algo/custom.py` is editable. The contract is `__init__(net, forward_op, ...)` builds the scheduler and stores hyperparameters, and `inference(observation, num_samples=1)` returns reconstructions of shape `(num_samples, C, H, W)`. A small set of per-problem hyperparameters is keyed by the `ENV` environment variable (`inv-scatter`, `blackhole`, `inpainting`), because the same conditioning rule may need different guidance strength across operators.

The starting scaffold returns random noise (no conditioning). Each method replaces this `Custom` class with its own conditioning rule.

```python
import torch
from tqdm import tqdm
from algo.base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler
import numpy as np


class Custom(Algo):
    """Custom algorithm for solving inverse problems with diffusion priors.

    Available utilities:
        - self.net(x, sigma): denoised estimate E[x_0 | x_t] (Tweedie); .img_channels, .img_resolution.
        - self.forward_op.forward(x): A(x).
        - self.forward_op.gradient(x, y, return_loss=True): (grad ||A(x)-y||^2, loss).
        - self.forward_op.loss(x, y): ||A(x)-y||^2, shape (batch,).
        - self.forward_op.device.
        - Scheduler(num_steps, schedule, timestep, scaling, sigma_max, sigma_min, ...):
              .sigma_steps, .factor_steps, .scaling_factor, .scaling_steps, .num_steps.
        - DiffusionSampler(scheduler).sample(model, x_start, SDE=False): unconditional sampling.
    """

    def __init__(self, net, forward_op,
                 diffusion_scheduler_config=None,
                 guidance_scale=10.0,
                 sde=True,
                 num_optim_steps=1000,
                 observation_weight=1.0,
                 base_lambda=0.25,
                 base_lr=0.5,
                 num_mc_samples=10,
                 **kwargs):
        super(Custom, self).__init__(net, forward_op)
        # Default: no algorithm. Store nothing.
        pass

    def inference(self, observation, num_samples=1, **kwargs):
        """Given y, return x with A(x) ~ y. Default: random noise (no conditioning)."""
        device = self.forward_op.device
        x = torch.randn(num_samples, self.net.img_channels,
                        self.net.img_resolution, self.net.img_resolution,
                        device=device)
        return x
```

## Evaluation settings

Three scientific inverse problems, keyed by `ENV`, single seed `42`:

1. **Inverse scattering** (`inv-scatter`) — recover permittivity from scattered EM fields through a differentiable scattering forward model. Metrics: PSNR, SSIM.
2. **Black hole imaging** (`blackhole`) — reconstruct an image from sparse EHT interferometric observations. Metrics: PSNR, blur-PSNR (f=15), closure-phase chi-squared (`cp_chi2`), closure-amplitude chi-squared (`camp_chi2`).
3. **FFHQ-256 inpainting** (`inpainting`) — recover a face from a box-masked observation with additive Gaussian noise (σ=0.05); the forward operator is a fixed pixel-wise mask. Metrics: PSNR, SSIM, LPIPS.

Higher PSNR / SSIM / blur-PSNR is better; lower LPIPS and chi-squared are better.
