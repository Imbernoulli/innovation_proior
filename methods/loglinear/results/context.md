## Research question

Score-based generative modeling learns a vector field `s_theta(x) ~= grad_x log p_data(x)` and
then turns that score into a sampler with Langevin dynamics:

```python
x_t = x_{t-1} + alpha * grad_x log p(x_{t-1}) + sqrt(2 * alpha) * z_t,   z_t ~ N(0, I).
```

A practical sampler works with a descending ladder of Gaussian-perturbed data distributions
`p_{sigma_1}, ..., p_{sigma_L}`, with `sigma_1 > sigma_2 > ... > sigma_L`, and runs from high noise
to low noise. The largest level must make exploration and score estimation tractable; the smallest
level must be close enough to the clean data distribution. The design question is how to place the
intermediate noise levels and how many to use.

## Background

**Scores and score matching.** For a density `p(x)`, the score is `grad_x log p(x)`. Hyvarinen
(2005) showed how to estimate scores without the normalizing constant by minimizing Fisher
divergence, equivalently an objective containing `tr(grad_x s_theta(x)) + (1/2)||s_theta(x)||^2`.
The trace term is expensive in high dimension, so scalable alternatives matter. Denoising score
matching (Vincent 2011) perturbs a data point by `q_sigma(x_tilde | x) = N(x, sigma^2 I)`, whose
conditional score is known:

```python
grad_{x_tilde} log q_sigma(x_tilde | x) = -(x_tilde - x) / sigma^2.
```

Sliced score matching (Song et al. 2019) estimates the trace with random projections and works on
the unperturbed distribution, at higher computational cost.

**Langevin dynamics.** Roberts and Tweedie (1996) and Welling and Teh (2011) give the sampling
primitive: use the score as a noisy gradient-ascent direction for `log p`. In implementations the
Metropolis correction is usually dropped, so step size and initialization matter.

**Annealing across intermediate distributions.** Simulated annealing (Kirkpatrick et al. 1983) and
annealed importance sampling (Neal 2001) use intermediate distributions to cross low-probability
barriers. In the score-based setting, the intermediate distributions are Gaussian-smoothed versions
of the data. A noise-conditional score network can share one model across all noise levels, and
annealed Langevin dynamics can carry samples from high noise to low noise.

**High-dimensional Gaussian geometry.** If `x ~ N(0, sigma^2 I)` in `R^D`, then
`||x||^2 / sigma^2 ~ chi2_D`. The radius `r = ||x||` concentrates in high dimension:

```python
r - sqrt(D) * sigma  ->  N(0, sigma^2 / 2).
```

Most mass is not near the center; it lies on a thin shell of radius about `sqrt(D) * sigma` and width
about `sigma / sqrt(2)`. The three-sigma rule says that a one-dimensional Gaussian puts almost all
of its mass within three standard deviations of its mean. These facts make it possible to reason
about handoffs between adjacent smoothed distributions through their radial high-density regions.

## Baselines

**A short fixed ladder copied across datasets.** Early multi-scale score samplers used a small fixed
number of descending noise levels between a large endpoint and a small endpoint, for example ten
levels from `1` to `0.01` on `[0, 1]` image data. This gives a working low-resolution recipe and
keeps sampling cost low.

**Uniform spacing in the noise value.** Another simple choice is `linspace(sigma_1, sigma_L, L)`.
It respects the endpoints and is easy to implement.

**Dense manual grids.** One can add many more intermediate levels and tune them by grid search. This
reduces the risk of a bad handoff and can be combined with a run of Langevin dynamics at each level.

## Evaluation settings

The natural yardsticks are image-generation datasets and metrics already used for score-based
generative models and neighboring image generators:

- Datasets: CIFAR-10 `32x32`, CelebA at `64x64`, LSUN categories at larger resolutions, and FFHQ
  resized to high resolution.
- Metrics: Fréchet Inception Distance (FID), with lower values better, and human-rating metrics
  such as HYPE when FID and visual quality disagree.
- Protocol: train one noise-conditional score network on all chosen noise levels using denoising
  score matching, then sample by annealed Langevin dynamics from the largest level to the smallest.
  A typical per-level step-size convention is proportional to `sigma_i^2`, so the Langevin update's
  signal-to-noise ratio is comparable across noise levels.

## Code framework

The schedule plugs into an existing annealed-sampling harness. The score network, Langevin update,
data pipeline, and metrics are already present. The open slot is a function that receives the number
of sampler intervals and the two endpoint noise levels, then returns a decreasing grid that the
sampler walks from high noise to low noise.

```python
import torch


def get_sigmas(n, t_min, t_max, device="cpu"):
    """Return n + 1 descending noise levels from t_max to t_min."""
    sigmas = torch.linspace(t_max, t_min, n + 1)
    # TODO: choose the intermediate noise levels.
    return sigmas.to(device)


def sample(score_net, x, n, t_min, t_max, device="cpu"):
    sigmas = get_sigmas(n, t_min, t_max, device)
    for i in range(n):
        sigma_cur, sigma_next = sigmas[i], sigmas[i + 1]
        x = score_net.step(x, sigma_cur, sigma_next)
    return x
```
