# Loss-Guided Diffusion (LGD-MC), distilled

LGD is a plug-and-play scheme for sampling from a posterior `p_0(x_0) · exp(-l_y(x_0)) / Z` using a
pretrained unconditional diffusion model as the prior and an arbitrary *differentiable* loss
`l_y(x_0)` (defined only on clean data) as the condition. The guidance term needed by the reverse
sampler, `∇_{x_t} log p_t(y | x_t)`, is an expectation of the clean-data likelihood over the intractable
denoising posterior `p(x_0 | x_t)`. The prior method DPS approximates that expectation by a single point
(the Tweedie/MMSE estimate `x_hat_t`), which mis-scales the guidance across the noise schedule. LGD-MC
replaces the point with a Gaussian surrogate `q = N(x_hat_t, r_t^2 I)` and estimates the expectation by
Monte Carlo over `n` samples while keeping a single backward pass through the diffusion network.

## Problem it solves

Plug-and-play conditional generation / inverse problems with a fixed pretrained diffusion prior: draw
samples respecting both the prior and a general differentiable loss, without retraining, without paired
`(x_0, y)` data, from a single network query per step (so fast samplers still apply), and with the
*scale* of the guidance approximately correct at every noise level.

## Key idea

The guidance likelihood factorizes (via `y ⊥ x_t | x_0`) as

```
p_t(y | x_t) = ∫ p(x_0 | x_t) p_0(y | x_0) dx_0 = E_{x_0 ~ p(x_0 | x_t)}[ exp(-l_y(x_0)) ].
```

DPS sets `p(x_0|x_t) ≈ δ(x_0 - x_hat_t)`, giving `∇_{x_t} log p_t(y|x_t) ≈ -∇_{x_t} l_y(x_hat_t)`. On a
tractable mixture-of-Gaussians the resulting guidance is **too large at high noise and too small at low
noise**, because the loss gradient is curved over the support of `p(x_0|x_t)` and evaluating it at the
posterior mean differs from averaging it over the (wide, at high noise) posterior.

LGD-MC keeps the single network call but uses a spread surrogate `q(x_0|x_t) = N(x_hat_t, r_t^2 I)`:

- **Mean `x_hat_t` (optimal):** maximizing `E_{p(x_0|x_t)}[log q]` over fixed-covariance Gaussians is
  `min_µ E‖µ - x_0‖^2`, whose argmin is `µ = E[x_0|x_t] = x_hat_t` — available from one denoiser call.
- **Width `r_t = σ_t / sqrt(1 + σ_t^2)` (posterior std):** the conjugate posterior variance of `x_0|x_t`
  for `x_t = x_0 + σ_t ε` with a unit-scale prior. `r_t → 1` (prior width) at high noise, `r_t → 0`
  (delta = DPS) at low noise.
- **Bias bound:** for `M = max_{x_0} p_0(y|x_0)`,
  `|E_p[p_0(y|x_0)] - E_q[p_0(y|x_0)]| ≤ 2M · TV(p(·|x_t), q(·|x_t))`. A delta is maximally far from a
  continuous spread-out posterior; whenever the Gaussian surrogate is closer in TV, this worst-case
  upper bound is smaller.

The guidance is the gradient of a **log-mean-exp of the negative loss** over `n` reparameterized samples
`x^(i) = x_hat_t + r_t ε^(i)`:

```
MC_n(x_t, y) = ∇_{x_t} log( (1/n) Σ_i exp(-l_y(x^(i))) ) = - Σ_i w_i ∇_{x_t} l_y(x^(i)),
               w_i = exp(-l_y(x^(i))) / Σ_j exp(-l_y(x^(j))).
```

The exact DPS formula is recovered when the Gaussian surrogate collapses to the delta at `x_hat_t`
(`r_t → 0`); with positive `r_t`, `n = 1` is a one-sample Gaussian estimate, not the DPS delta. Because
all `n` samples branch off the *same* `x_hat_t = D_θ(x_t, t)`, only one backward pass through the
diffusion network is needed: aggregate the cheap per-sample loss gradients, then use one
vector-Jacobian product through `D_θ`.

For an arbitrary differentiable loss, the aggregate cotangent should use the softmin weights above.
The canonical InverseBench implementation supplied here uses the least-squares interface exposed by
`forward_op.gradient`: it averages the per-sample squared-residual gradients and then applies the
DPS root-loss normalization. That is a code-level mean-gradient realization of the sampled
least-squares guidance, not an explicit softmin-weighted implementation of the general log-mean-exp
formula for arbitrary losses.

## Stabilization and scale

- **Root-loss normalization (inherited from DPS):** with `loss = ‖A(x) - y‖^2`, multiplying its
  gradient by `0.5 / sqrt(loss)` yields `∇ sqrt(loss) = ∇‖A(x) - y‖`, the stable residual-norm gradient
  (DPS's `ζ_i = ζ' / ‖y - A(x_hat)‖`).
- **Overall `guidance_scale`:** a classifier-guidance-style multiplier
  (`s · ∇ log p(y|x) = ∇ log(p(y|x)^s / Z)` sharpens toward the loss's modes); tuned per problem since
  its natural value tracks the forward operator's loss scale.
- The guidance is **subtracted** from the reverse step (minimizing the loss = maximizing the likelihood).

## Final algorithm (per reverse step `i`)

```
x_cur ← detach(x_next), requires_grad
σ, factor, scaling_factor, scaling ← scheduler[i];   r_t ← σ / sqrt(1 + σ^2)
denoised ← net(x_cur / scaling, σ)                       # ONE call: x_hat_t = E[x0|x_t]
samples  ← denoised + r_t · ε,  ε ~ N(0, I) of shape (n, ...)   # q = N(x_hat_t, r_t^2 I)
grad_i, loss_i ← forward_op.gradient(samples, y)         # ∇||A(x^(i)) - y||^2, ||A(x^(i)) - y||^2
avg_grad ← mean_i grad_i (detached)
ll_grad  ← autograd.grad(denoised, x_cur, avg_grad)      # one VJP through the denoiser
ll_grad  ← ll_grad · 0.5 / sqrt(avg_loss)                # → root-loss gradient (stable)
score    ← (denoised - x_cur / scaling) / σ^2 / scaling  # Tweedie score, EDM scaled form
if SDE: x_next ← x_cur·scaling_factor + factor·score + sqrt(factor)·ε
else:   x_next ← x_cur·scaling_factor + 0.5·factor·score
x_next   ← x_next - ll_grad · guidance_scale
```

The EDM scheduler precomputes `scaling_factor = 1 - (ṡ/s)Δt`, `factor = 2 s^2 σ̇ σ Δt`; VP uses
`σ(t) = sqrt(exp(½β_d t^2 + β_min t) - 1)`, `s(t) = 1/sqrt(exp(½β_d t^2 + β_min t))`, `β_d = 19.9`,
`β_min = 0.1`.

## Relation to prior methods

- **DPS** is the delta-surrogate point estimate at `x_hat_t`; LGD-MC approaches that behavior as
  `r_t → 0`, while positive `r_t` keeps posterior spread.
- **Classifier guidance** supplies `∇ log p_t(y|x_t)` from a *trained* noisy classifier (paired data,
  not plug-and-play); LGD derives it from the loss with no training. LGD reuses CG's gradient-scale knob.
- **D-PnP** optimizes the prior to a slow point estimate; LGD is a sampler that inherits fast diffusion
  solvers.

## Working code

Faithful to the InverseBench implementation, written as the `LGD` class. The expensive
backward through `net` happens once per step; `num_samples` only multiplies forward-operator gradients.

```python
import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler
import numpy as np

import wandb


class LGD(Algo):
    def __init__(self,
                 net,
                 forward_op,
                 diffusion_scheduler_config,
                 guidance_scale,
                 num_samples=10,
                 batch_grad=True,
                 sde=True):
        super(LGD, self).__init__(net, forward_op)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sde = sde
        self.num_samples = num_samples
        self.batch_grad = batch_grad

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        x_initial = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution,
                                self.net.img_resolution, device=device) * self.scheduler.sigma_max
        x_next = x_initial
        x_next.requires_grad = True
        pbar = tqdm(range(self.scheduler.num_steps))

        for i in pbar:
            x_cur = x_next.detach().requires_grad_(True)

            sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], \
                self.scheduler.scaling_factor[i]
            rt = sigma / np.sqrt(1 + sigma ** 2)

            denoised = self.net(x_cur / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device))

            samples = denoised + torch.randn((self.num_samples, *denoised.shape[1:]), device=device) * rt

            if self.batch_grad:
                gradient, loss_scale = self.forward_op.gradient(samples, observation, return_loss=True)
                gradients = gradient
                avg_loss = loss_scale
            else:
                gradients = torch.empty((self.num_samples, *denoised.shape[1:]), device=device)
                losses = np.empty(self.num_samples)
                for j in range(self.num_samples):
                    gradient, loss_scale = self.forward_op.gradient(samples[j:j+1], observation, return_loss=True)
                    gradients[j] = gradient
                    losses[j] = loss_scale
                avg_loss = losses.mean()

            avg_grad = torch.mean(gradients, dim=0, keepdim=True).detach()

            ll_grad = torch.autograd.grad(denoised, x_cur, avg_grad)[0]
            ll_grad = ll_grad * 0.5 / torch.sqrt(avg_loss)

            score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
            pbar.set_description(
                f'Iteration {i + 1}/{self.scheduler.num_steps}. '
                f'Data fitting loss: {torch.sqrt(loss_scale)}')
            if wandb.run is not None:
                wandb.log({'data_fitting_loss': torch.sqrt(loss_scale)})

            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5

            x_next -= ll_grad * self.scale

        return x_next
```
