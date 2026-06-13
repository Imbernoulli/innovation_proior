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

The precise goal is a single sampler that (1) draws from `p(x_0 | y)` using only the
pre-trained unconditional score plus the known forward model, with no retraining per task;
(2) is correct in the presence of **measurement noise** `n ≠ 0`, not just the idealized
noiseless case; (3) works for a **general** forward operator — including **nonlinear** `A` —
without requiring a special algebraic structure such as an SVD or an easily-defined
projection; (4) needs little task-specific tuning. The existing diffusion-based solvers below
each achieve a subset; none achieves all four at once, and in particular the noise and
nonlinearity requirements are where they break. Closing that gap is the problem.

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
drift `grad_{x_t} log p_t(y | x_t)`. The first term is the pre-trained score; the catch is the
second. There is an explicit measurement model only between `y` and the *clean* `x_0`
(`p(y | x_0) = N(y | A(x_0), sigma^2 I)` in the Gaussian case), whereas the reverse process
needs the likelihood at the *noised* iterate `x_t`. In the probabilistic graph
`y <- x_0 -> x_t`, the arrow `x_0 -> y` and `x_0 -> x_t` are tractable, but there is no direct
`x_t -> y` edge: `p_t(y | x_t)` has no closed form because `y` only depends on `x_t` through
the unknown `x_0`. This time-level likelihood is the crux.

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

**The diagnostic failure mode that motivates all of this.** It is well documented that
diffusion solvers built on enforcing *exact* measurement consistency degrade sharply once the
measurement is noisy: enforcing `A x = y` on a corrupted `y` overfits the noise, and for
operators like super-resolution or deblurring the transpose `A^T` applied during a consistency
step *amplifies* that noise, so the reconstruction accumulates error and drifts to a wrong
solution. This is the phenomenon any noise-robust solver has to avoid reproducing.

## Baselines

These are the prior diffusion-based inverse-problem solvers a general noisy solver has to improve on.

**Projection-onto-measurement-subspace solvers (Song et al. 2021 score-SDE; ILVR, Choi et al.
2021; Chung et al. 2022b).** Drop the likelihood-score term entirely: take an unconditional
reverse-diffusion step, then **project** the iterate onto the measurement set
`C = { x : A x = y }` (a POCS / alternating-projection step), assuming `n ≈ 0`. Clean and
training-free, and strong on *noiseless* linear problems. **Gap:** the projection imposes
perfect data consistency, so when `n ≠ 0` it forces the sample to agree with corrupted
measurements and the noise is amplified (the `A^T` in the projection step), as in the
diagnostic failure above; and a projection onto `{ A x = y }` presupposes a linear, easily
projectable `A` — it does not extend naturally to nonlinear forward operators.

**Spectral-domain solvers — SNIPS / DDRM (Kawar et al. 2021; 2022).** Run the diffusion in the
basis given by the SVD of `A`, which lets the measurement-domain Gaussian noise be tied to
spectral-domain noise and handled in closed form; this does cope with noise. **Gap:** it
requires an explicit, cheap SVD of the forward operator. For complex operators the SVD is
expensive or infeasible, restricting the family to cases like separable blur kernels; general
or nonlinear `A` is out of reach.

**Annealed linear-likelihood Langevin — robust-CSGM (Jalal et al. 2021).** For linear
`A(x) = A x` and Gaussian noise, use the closed-form `t=0` likelihood score
`grad_{x} log p(y | x) = A^H (y - A x) / sigma^2` directly inside annealed Langevin dynamics.
**Gap:** that expression is the likelihood score of the *clean* model; it is exact only at
`t = 0` and wrong at every other noise level actually used in the reverse process. The
denominator is patched by hand with
`A^H (y - A x) / (sigma^2 + gamma_t^2)` with `gamma_t -> 0` a decreasing hyperparameter
sequence — a heuristic correction, and still linear-only.

**Manifold constrained gradient — MCG (Chung et al. 2022a).** The closest ancestor. Use the
Tweedie estimate `x_0_hat` and step along the gradient of a data-fidelity term evaluated at
it, `grad_{x_i} || W (y - A x_0_hat) ||_2^2`. Its central result (Theorem 1) is geometric: in
the diffusion setting a single denoising step acts like an orthogonal projection onto the data
manifold `M`, the score only resolves the direction normal to `M`, and the data-fidelity
gradient through `x_0_hat` equals the projection of the fidelity term onto the tangent space
`T_{x_0_hat} M` — i.e. it moves *along* the manifold to use the measurement to discriminate
points the score cannot. After that gradient step MCG **additionally projects** the iterate
onto the measurement subspace `C` to enforce data consistency. **Gap:** the extra projection
again assumes the noiseless regime; with noisy `y` it overshoots data consistency, pushes the
sample off the manifold, and accumulates error (the same noise-amplification pathology), so it
degrades on noisy problems; and choosing the weighting `W` per application is a heuristic.

**Tweedie coarse-to-fine gradient (Kadkhodaie & Simoncelli 2021).** Uses a likelihood gradient
obtained from the Tweedie-denoised estimate in a coarse-to-fine schedule. **Gap:** framed for
specific linear restoration tasks; not a general noisy/nonlinear solver.

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
unconditional reverse step (the discretized reverse SDE / PF-ODE) is given. What is *not*
settled is how — if at all — to bend each reverse step toward the measurement `y`; that
conditioning rule is the single empty slot below.

```python
import torch
import numpy as np
from .base import Algo
from utils.scheduler import Scheduler


class Solver(Algo):
    """Sample x given an observation y, using a pre-trained diffusion prior `net`
    (returns the Tweedie posterior mean E[x_0 | x_t]) and a known forward
    operator `forward_op`. The reverse-diffusion harness already exists; the
    measurement-conditioning rule does not."""

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
            #       the in-loop denoised estimate, bend this reverse step toward y.
            pass

        return x_next
```

The harness supplies, at every reverse step, the noised iterate, its Tweedie denoised
estimate, and the unconditional update; the `# TODO` is the one place the conditioning on `y`
will live.
