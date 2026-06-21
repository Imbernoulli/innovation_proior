# Context: plug-and-play conditional generation with pretrained diffusion priors (circa 2022-2023)

## Research question

A diffusion model trained on a signal distribution `p_0(x_0)` is a rich, reusable prior. The
practical question is how to *reuse* it for controllable generation without retraining: given a
pretrained unconditional model and a user-supplied, differentiable loss `l_y(x_0)` that scores how
well a clean signal `x_0` satisfies some condition `y` (a measurement-consistency term for an inverse
problem, a classifier log-probability, a CLIP matching score, a path/obstacle penalty for motion),
draw samples from the *posterior*

```
p^(l)_0(x_0 | y)  =  p_0(x_0) · exp(-l_y(x_0)) / Z,        Z = ∫ p_0(x_0) exp(-l_y(x_0)) dx_0.
```

This is "plug-and-play": the prior and the loss are decoupled, `y` is known only at test time, and the
loss can be anything differentiable — not just a linear least-squares term, and not something the prior
was trained to know about. A diffusion sampler integrates a reverse process driven by the *score* of
`p_t(x_t)` at every noise level `t`; to bend it toward the posterior we need the corresponding
conditional score at every noise level. By Bayes' rule that splits into the unconditional score (which
the pretrained model gives us) plus a *guidance term* `∇_{x_t} log p_t(y | x_t)`.

## Background

By this time diffusion models are the dominant generative-modeling framework for images, audio, 3D,
motion, and scientific signals. The load-bearing concepts a guidance method rests on:

- **The forward noising process and the score.** A data distribution `p_0(x_0)` induces a family of
  noised marginals `p_t(x_t)` by adding i.i.d. Gaussian noise, `x_t = x_0 + σ_t ε`, `ε ~ N(0, I)`,
  with `σ_t` increasing in `t`. The object a diffusion model learns is the score `∇_{x_t} log p_t(x_t)`,
  trained by denoising score matching (Vincent 2011): minimizing
  `E_{x_0, x_t} ‖D_θ(x_t, t) - x_0‖^2` makes the network `D_θ` the minimum-mean-squared-error denoiser.

- **Tweedie's formula (Efron 2011) — denoiser ⇔ score.** The optimal MMSE denoiser and the score are
  two views of the same object: `x_hat_t := D_θ(x_t, t) = E[x_0 | x_t] = x_t + σ_t^2 ∇_{x_t} log p_t(x_t)`.
  Equivalently `∇_{x_t} log p_t(x_t) = (D_θ(x_t, t) - x_t) / σ_t^2`. So a single network call yields both
  the score (for the sampler) and the posterior-mean estimate of the clean signal `x_hat_t` (a natural
  hook for any quantity that is only defined on clean data).

- **The reverse process as an SDE/ODE (the sampler).** The marginals `p_t` are tied by a stochastic
  differential equation whose drift is the score; integrating it backward from noise to data generates
  samples. In the EDM common framework (Karras et al. 2022) the probability-flow ODE is
  `dx = -σ̇(t) σ(t) ∇_x log p(x; σ(t)) dt`, with a deterministic (ODE) or stochastic (Langevin/SDE)
  variant, and a time-dependent scaling `s(t)` of the variable giving the generalized form
  `dx = [ (ṡ/s) x - s^2 σ̇ σ ∇_x log p(x/s; σ) ] dt` (+ noise for the SDE). The variance-preserving (VP)
  choice uses `σ(t) = sqrt(exp(½ β_d t^2 + β_min t) - 1)` and `s(t) = 1/sqrt(exp(½ β_d t^2 + β_min t))`
  with `β_d = 19.9`, `β_min = 0.1`. A wave of fast first-order ODE/SDE solvers (DDIM and successors)
  cut the number of network evaluations from ~1000 to ~10-100.

- **The guidance term is an expectation over an intractable posterior.** The condition `y` is a
  function of the *clean* signal: in the graphical model `x_0 → x_t` and `x_0 → y`, with `y` and `x_t`
  conditionally independent given `x_0`. Therefore the guidance likelihood at noise level `t` is

  ```
  p_t(y | x_t) = ∫ p(x_0 | x_t) p_0(y | x_0) dx_0  =  E_{x_0 ~ p(x_0 | x_t)}[ p_0(y | x_0) ],
  ```

  an expectation of the clean-data likelihood `p_0(y | x_0) ∝ exp(-l_y(x_0))` over the *denoising
  posterior* `p(x_0 | x_t)`. That posterior is exact: evaluating or sampling it accurately needs many
  diffusion steps, so the integral is not directly tractable from a single network call.

- **A tractable one-dimensional diagnostic.** On a tractable one-dimensional toy — `p_0` a mixture of
  two well-separated Gaussians, `y` the mixture label, so `p_t(x_t)` is itself a Gaussian mixture and
  the true `∇_{x_t} log p_t(y | x_t)` is computable in closed form at any noise level — one can directly
  compare the closed-form guidance to any approximation.

## Baselines

The prior methods a new guidance scheme is measured against and reacts to.

**Diffusion Posterior Sampling (DPS) — Chung et al., ICLR 2023.** The most general existing
plug-and-play guidance for diffusion priors. DPS confronts the intractable integral
`p_t(y | x_t) = E_{x_0 ~ p(x_0 | x_t)}[p_0(y | x_0)]` by collapsing the denoising posterior to a single
point: it replaces `p(x_0 | x_t)` with a delta mass at the MMSE estimate `x_hat_t = D_θ(x_t, t)`, giving

```
p_t(y | x_t) ≈ p_0(y | x_hat_t),     ∇_{x_t} log p_t(y | x_t) ≈ ∇_{x_t} log p_0(y | x_hat_t) = -∇_{x_t} l_y(x_hat_t),
```

a backprop of the clean-data loss gradient through the denoiser to `x_t` (the normalizer `Z` is
`x_t`-independent and drops out). For a Gaussian measurement this is a negative squared-residual
gradient through the denoiser; DPS writes the update as `-ρ ∇_{x_t} ‖y - A(x_hat_t)‖^2` with step scale
`ρ = 1/σ^2`. In its discrete implementation DPS does not use that raw `ρ`; instead it uses the
empirically stabilizing rule
`ζ_i = ζ' / ‖y - A(x_hat(x_i))‖` with `ζ'` a constant — i.e. it normalizes the squared-residual
gradient by the residual norm, which is the gradient of the *root* loss `‖y - A(x_hat)‖` since
`∇‖·‖ = ∇‖·‖^2 / (2‖·‖)`. DPS works on nonlinear inverse problems and noisy measurements, where it is
analyzed via a Jensen gap that shrinks with measurement noise.

**Classifier guidance (CG) — Dhariwal & Nichol, NeurIPS 2021.** Obtain the conditional score by Bayes'
rule, `∇_{x_t} log p_t(x_t | y) = ∇_{x_t} log p_t(x_t) + ∇_{x_t} log p_t(y | x_t)`, and supply the
second term from a *separately trained* classifier `p_φ(y | x_t)` that operates on *noisy* images at
every level. An optional gradient scale `s` multiplies the guidance; since
`s · ∇_x log p(y | x) = ∇_x log (p(y | x)^s / Z)`, raising `s` sharpens the conditional toward the
classifier's modes, trading diversity for fidelity. It requires paired `(x_0, y)` training data to fit
the noisy classifier, and a new classifier for each family of conditions.

**Diffusion models as plug-and-play priors (D-PnP) — Graikos et al., NeurIPS 2022.** Use the diffusion
model as a learned prior inside a stochastic-optimization objective and *optimize* for an `x_0` that is
consistent with both the prior and the loss, finding a point estimate.

**Reconstruction guidance / linear plug-and-play solvers.** A family of methods handles *linear*
inverse problems with least-squares losses in closed form by exploiting the Gaussianity of
`p(x_0 | x_t)` (e.g. pseudo-inverse-based and least-squares-guidance schemes). They are specialized
to linear operators and Gaussian/least-squares losses.

## Evaluation settings

The natural yardsticks already in use for plug-and-play conditional generation, all using pretrained
unconditional diffusion models:

- **A tractable one-dimensional mixture of Gaussians** with a logistic/label loss, where `p_0`, `p_t`,
  and the true guidance `∇_{x_t} log p_t(y | x_t)` are all computable in closed form. Used to compare an
  approximate guidance term directly against ground truth at fixed noise levels (e.g. `σ_t = 1` and
  `σ_t = 80`) and to score sample quality by KL divergence between the generated marginal and the target
  Gaussian. The diagnostic harness, not a benchmark.
- **Image super-resolution** (e.g. 64×64 → 256×256, bicubic downsampling) on ImageNet validation
  images, a linear inverse problem with loss `l_y(x_0) = ‖y - H x_0‖^2 / (2 s_t^2)`; metrics FID and a
  pretrained classifier's accuracy on the reconstructions; samplers of ~100 steps (DDIM / DDPM).
- **Conditional image generation** on ImageNet 256×256 with a pretrained unconditional model, guided by
  a classifier log-probability or a CLIP matching score (with the standard data-augmentation tricks for
  CLIP guidance); metrics FID and classifier accuracy; fast samplers (e.g. stochastic Heun).
- **Controllable human-motion synthesis** with a pretrained text-to-motion diffusion model, guided by a
  path-following distance loss or an obstacle-avoidance collision penalty
  `Σ_i sigmoid(-(‖root(x_0) - y_obs‖^2 - 1) · 50) · 100`; a 100-step DDIM sampler; metrics are the
  objective loss (normalized per frame) and a text/motion embedding distance.
- **Scientific inverse problems** with a fixed pretrained denoiser and fixed forward operators: an
  optical-tomography inverse-scattering problem (permittivity from scattered fields; PSNR/SSIM), a
  radio-astronomy black-hole imaging problem (image from sparse interferometric data; PSNR, blur-PSNR,
  closure-phase chi-squared), and FFHQ-256 image inpainting under a fixed box mask with additive
  Gaussian noise `σ = 0.05` (PSNR/SSIM/LPIPS). The denoiser, the forward operators, and the evaluation
  problems are fixed; only the guidance algorithm varies.

## Code framework

The guidance method plugs into the same diffusion-inference harness already used for the baselines. The
pretrained denoiser, the forward operator, and the noise schedule all exist; what does not yet exist is
*how to turn the clean-data loss into a per-step update on the noisy iterate* — that is the single empty
slot. The substrate:

- `net(x, sigma)` returns the MMSE denoised estimate `E[x_0 | x_t]` (Tweedie) for one network call.
- `forward_op.forward(x)` computes `A(x)`; `forward_op.gradient(x, y, return_loss=True)` returns
  `(∇_x ‖A(x) - y‖^2, ‖A(x) - y‖^2)`; `forward_op.loss(x, y)` returns `‖A(x) - y‖^2`.
- `Scheduler(num_steps, schedule, timestep, scaling)` precomputes, per step `i`, `sigma_steps[i]`,
  `scaling_steps[i]`, `scaling_factor[i] = 1 - (ṡ/s)Δt`, and `factor_steps[i] = 2 s^2 σ̇ σ Δt` for the
  EDM discretized reverse SDE/ODE.
- The unconditional reverse step is the standard EDM update `x_next = x_cur·scaling_factor +
  factor·score (+ sqrt(factor)·ε for the SDE)`, with `score = (denoised - x_cur/scaling)/σ^2/scaling`.

```python
import numpy as np
import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler


class Custom(Algo):
    """Guided diffusion with a pretrained denoiser as the prior and a differentiable
    measurement loss supplied by `forward_op`."""

    def __init__(self, net, forward_op, diffusion_scheduler_config,
                 guidance_scale, **kwargs):
        super().__init__(net, forward_op)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**self.diffusion_scheduler_config)
        self.sde = True
        # TODO: any state the loss-to-update rule needs.

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        x_next = torch.randn(num_samples, self.net.img_channels,
                             self.net.img_resolution, self.net.img_resolution,
                             device=device) * self.scheduler.sigma_max
        x_next.requires_grad = True

        for i in tqdm(range(self.scheduler.num_steps)):
            x_cur = x_next.detach().requires_grad_(True)
            sigma = self.scheduler.sigma_steps[i]
            factor = self.scheduler.factor_steps[i]
            scaling_factor = self.scheduler.scaling_factor[i]
            scaling = self.scheduler.scaling_steps[i]

            denoised = self.net(x_cur / scaling, torch.as_tensor(sigma).to(device))

            # TODO: turn the clean-data loss into a loss-gradient update on x_cur,
            # using the already-computed denoised estimate.
            loss_update = None

            # unconditional reverse step (EDM SDE/ODE)
            score = (denoised - x_cur / scaling) / sigma ** 2 / scaling
            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5

            x_next = x_next - loss_update * self.scale

        return x_next
```
