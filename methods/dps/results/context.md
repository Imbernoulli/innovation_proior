# Context: solving noisy inverse problems with a pre-trained diffusion prior

## Research question

We have a pre-trained diffusion model that has learned the prior `p(x)` over some signal
distribution — natural images or other structured signals — encoded
as a time-dependent score network `s_theta(x_t, t) ~ grad_{x_t} log p_t(x_t)`. Separately we
have an inverse problem: a measurement

```
y = A(x_0) + n
```

where `A` is a known forward operator (a mask, a blur, a downsampler, or a Fourier-magnitude
map) and `n` is detector noise, taken
white Gaussian `n ~ N(0, sigma^2 I)` in the canonical case. We want to recover `x_0`. Because
`A` is many-to-one the problem is ill-posed, so the right object is not a single inverse but
the **posterior** `p(x_0 | y) ∝ p(y | x_0) p(x_0)`: use the diffusion prior as `p(x_0)` and
condition it on the data.

The question is how to build a sampler that draws from `p(x_0 | y)` using only the
pre-trained unconditional score plus the known forward model — across measurement noise levels
`n` (Gaussian or signal-dependent) and forward operators `A` that may be linear or nonlinear.
The existing diffusion-based solvers below are the starting point.

## Background

**The reverse-SDE / score-based generative frame (Song et al. 2021; Anderson 1982).** A
diffusion model defines a forward noising SDE `dx = -(beta(t)/2) x dt + sqrt(beta(t)) dw`
that carries data `x(0) ~ p_data` to an isotropic Gaussian `x(T) ~ N(0, I)`. Generation runs
the corresponding reverse SDE

```
dx = [ -(beta(t)/2) x - beta(t) grad_{x_t} log p_t(x_t) ] dt + sqrt(beta(t)) dw_bar,
```

whose only unknown is the time-dependent **score** `grad_{x_t} log p_t(x_t)`, approximated by
a network `s_theta` trained with denoising score matching (Vincent 2011). Discretizing and
integrating this reverse SDE samples `p(x)`. In the variance-preserving (VP) form, equivalent
to DDPM (Ho et al. 2020), the forward marginal is closed-form Gaussian

```
x_t = sqrt(alpha_bar(t)) x_0 + sqrt(1 - alpha_bar(t)) z,   z ~ N(0, I).
```

**The Bayesian/posterior frame.** To condition on `y` one wants to sample the posterior. The
score of the time-marginal posterior splits cleanly by Bayes' rule:

```
grad_{x_t} log p_t(x_t | y) = grad_{x_t} log p_t(x_t) + grad_{x_t} log p_t(y | x_t),
```

so the conditional reverse SDE is the unconditional one with an extra **likelihood-score**
drift `grad_{x_t} log p_t(y | x_t)`. The first term is the pre-trained score. The measurement
model is given explicitly between `y` and the *clean* `x_0`
(`p(y | x_0) = N(y | A(x_0), sigma^2 I)` in the Gaussian case), whereas the reverse process
needs the likelihood at the *noised* iterate `x_t`. In the probabilistic graph
`y <- x_0 -> x_t`, the arrows `x_0 -> y` and `x_0 -> x_t` are tractable, and `y` depends on
`x_t` only through `x_0`.

**Tweedie's formula — the posterior mean of `x_0` given `x_t`.** A classical empirical-Bayes
result (Robbins 1956; Stein 1981; Efron 2011): when `x_t | x_0` is Gaussian as above, the MMSE
estimate of the clean signal is a closed-form function of the score,

```
x_0_hat := E[x_0 | x_t] = (1 / sqrt(alpha_bar(t))) ( x_t + (1 - alpha_bar(t)) grad_{x_t} log p_t(x_t) ),
```

so plugging in `s_theta` for the score gives an in-the-loop denoised estimate of `x_0` at any
noise level, for free. Tweedie follows from the exponential-family identity: for
`p(y | eta) = p_0(y) exp(eta^T T(y) - phi(eta))`, the posterior mean satisfies
`(grad_y T(y))^T E[eta | y] = grad_y log p(y) - grad_y log p_0(y)`; specializing to the
Gaussian `p(x_t | x_0)` reproduces the formula above.

**Jensen gap.** For a (possibly non-convex) `f` and random `x`, the Jensen gap is
`J(f, x~p) = E[f(x)] - f(E[x])`; it can be bounded whenever `f` is Hölder/Lipschitz around the
mean (Gao et al. 2017), e.g. `|E[f(X)] - f(E[X])| <= K E[||X - mu||^alpha]` if
`|f(x) - f(mu)| <= K ||x - mu||^alpha`. The relevant Lipschitz fact: for a `q`-dimensional
isotropic Gaussian density `h_sigma(u) = N(y; u, sigma^2 I)` viewed as a normalized density in
its mean argument,
`|h_sigma(u) - h_sigma(v)| <= L_sigma ||u - v||`. Since
`||grad_u h_sigma(u)|| = (||u-y||/sigma^2)(2 pi sigma^2)^(-q/2)
exp(-||u-y||^2/(2 sigma^2))`, the Euclidean gradient bound is attained at
`||u-y|| = sigma` and equals
`L_sigma = e^{-1/2} (2 pi)^(-q/2) sigma^(-(q+1))`. Norm or coordinatewise variants can add
dimension factors, but for the normalized `q`-dimensional density the noise-scale behavior is
the same: the constant grows as `sigma -> 0` and tends to zero as `sigma -> infinity`.

## Baselines

These are the prior diffusion-based inverse-problem solvers in use.

**Projection-onto-measurement-subspace solvers (Song et al. 2021 score-SDE; ILVR, Choi et al.
2021; Chung et al. 2022b).** Drop the likelihood-score term entirely: take an unconditional
reverse-diffusion step, then **project** the iterate onto the measurement set
`C = { x : A x = y }` (a POCS / alternating-projection step), for the regime `n ≈ 0`.
Training-free, and used on noiseless linear problems.

**Spectral-domain solvers — SNIPS / DDRM (Kawar et al. 2021; 2022).** Run the diffusion in the
basis given by the SVD of `A`, which lets the measurement-domain Gaussian noise be tied to
spectral-domain noise and handled in closed form, so it handles noise. Applied to operators
with an explicit, cheap SVD such as separable blur kernels.

**Annealed linear-likelihood Langevin — robust-CSGM (Jalal et al. 2021).** For linear
`A(x) = A x` and Gaussian noise, use the closed-form `t=0` likelihood score
`grad_{x} log p(y | x) = A^H (y - A x) / sigma^2` directly inside annealed Langevin dynamics.
The denominator is annealed across the trajectory with
`A^H (y - A x) / (sigma^2 + gamma_t^2)`, where `gamma_t -> 0` is a decreasing hyperparameter
sequence.

**Manifold constrained gradient — MCG (Chung et al. 2022a).** Use the
Tweedie estimate `x_0_hat` and step along the gradient of a data-fidelity term evaluated at
it, `grad_{x_i} || W (y - A x_0_hat) ||_2^2`. Its central result (Theorem 1) is geometric: in
the diffusion setting a single denoising step acts like an orthogonal projection onto the data
manifold `M`, the score only resolves the direction normal to `M`, and the data-fidelity
gradient through `x_0_hat` equals the projection of the fidelity term onto the tangent space
`T_{x_0_hat} M` — i.e. it moves *along* the manifold to use the measurement to discriminate
points the score cannot. After that gradient step MCG **additionally projects** the iterate
onto the measurement subspace `C` to enforce data consistency. The weighting `W` is set per
application.

**Tweedie coarse-to-fine gradient (Kadkhodaie & Simoncelli 2021).** Uses a likelihood gradient
obtained from the Tweedie-denoised estimate in a coarse-to-fine schedule, framed for
specific linear restoration tasks.

## Evaluation settings

The natural yardsticks for an image-domain diffusion inverse-problem solver at the time:

- **Datasets / priors.** FFHQ `256x256` faces (Karras et al. 2019) and ImageNet `256x256`
  (Deng et al. 2009), 1k held-out validation images each, with pre-trained unconditional score
  networks (ADM / guided-diffusion checkpoints, Dhariwal & Nichol 2021), images normalized to
  `[0,1]`. The same score checkpoint is shared across all diffusion-based methods for a fair
  comparison.
- **Linear forward operators.** Box inpainting (`128x128` masked region) and random inpainting
  (≈92% of pixels masked, all RGB channels); `4x` super-resolution by bicubic downsampling;
  Gaussian blur (`61x61` kernel, std `3.0`) and motion blur (`61x61`, intensity `0.5`).
- **Nonlinear forward operators.** Fourier phase retrieval (measure only the Fourier
  magnitude `| F x |`, typically with oversampling for uniqueness) and non-uniform / nonlinear
  deblurring through a neural-network-distilled forward model.
- **Noise models.** Additive white Gaussian on the measurement, `sigma = 0.05` (on the
  `[0,1]` scale); and signal-dependent Poisson / shot noise, `lambda = 1.0`.
- **Metrics.** Perceptual FID and LPIPS as the primary scores, with PSNR / SSIM as standard
  distortion metrics; lower FID/LPIPS and higher PSNR/SSIM are better. Sampling cost is
  measured in number of function evaluations (NFE) and wall-clock time.

## Code framework

The sampler plugs into the EDM-style diffusion scaffold that already exists: a `Scheduler`
that precomputes the noise levels and the per-step coefficients of the reverse update, a
pre-trained denoiser/score `net` that returns the Tweedie posterior mean `E[x_0 | x_t]`, and a
`forward_op` exposing the known measurement operator together with its gradient. The
unconditional reverse step (the discretized reverse SDE / PF-ODE) is given. The open slot is
the rule that conditions each reverse step on the measurement `y`.

```python
import torch
import numpy as np
from .base import Algo
from utils.scheduler import Scheduler


class Solver(Algo):
    """Sample x given an observation y, using a pre-trained diffusion prior `net`
    (returns the Tweedie posterior mean E[x_0 | x_t]) and a known forward
    operator `forward_op`. The reverse-diffusion harness already exists; the
    measurement-conditioning rule is the open slot."""

    def __init__(self, net, forward_op, diffusion_scheduler_config, guidance_scale, sde=True):
        super().__init__(net, forward_op)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sde = sde

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
        # start from pure noise at the maximum noise level
        x_initial = torch.randn(num_samples, self.net.img_channels,
                                self.net.img_resolution, self.net.img_resolution,
                                device=device) * self.scheduler.sigma_max
        x_next = x_initial
        x_next.requires_grad = True

        for i in range(self.scheduler.num_steps):
            x_cur = x_next.detach().requires_grad_(True)
            sigma = self.scheduler.sigma_steps[i]
            factor = self.scheduler.factor_steps[i]
            scaling_factor = self.scheduler.scaling_factor[i]

            # Tweedie posterior mean E[x_0 | x_t] from the pre-trained prior
            denoised = self.net(x_cur / self.scheduler.scaling_steps[i],
                                torch.as_tensor(sigma).to(x_cur.device))

            # unconditional reverse step (discretized reverse SDE / PF-ODE)
            score = ((denoised - x_cur / self.scheduler.scaling_steps[i])
                     / sigma ** 2 / self.scheduler.scaling_steps[i])
            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5

            # TODO: choose the measurement-conditioning rule.
            #       Given the observation, the forward operator forward_op, and
            #       the in-loop denoised estimate, condition this reverse step on y.
            pass

        return x_next
```

The harness supplies, at every reverse step, the noised iterate, its Tweedie denoised
estimate, and the unconditional update; the `# TODO` is where the conditioning on `y` lives.
