# Log-linear (geometric) noise schedule

The schedule places the noise levels of an annealed score-based sampler at equal spacing in
`log(sigma)`. Equivalently, the adjacent ratio
`gamma = sigma_{i-1} / sigma_i` is constant as the sampler moves from the largest noise level to the
smallest.

## Problem

Annealed score-based sampling uses a ladder of Gaussian-perturbed data distributions
`p_{sigma_1}, ..., p_{sigma_L}` with `sigma_1 > ... > sigma_L`. Each level is initialized from the
previous level's output. If adjacent levels are too far apart, the next score field is queried in a
weakly supported region and the warm start fails; if they are too close, sampler work is wasted.

## Derivation

Use the one-point approximation `p_sigma(x) = N(0, sigma^2 I)` in `R^D`. For
`r = ||x||`,

```text
||x||^2 / sigma^2 ~ chi2_D,
r - sqrt(D) sigma -> N(0, sigma^2 / 2).
```

So the mass lies on a thin shell with radial approximation
`r_i ~ N(m_i, s_i^2)`, where `m_i = sqrt(D) sigma_i` and `s_i^2 = sigma_i^2 / 2`.
The high-density radial band of the previous level is
`I_{i-1} = [m_{i-1} - 3 s_{i-1}, m_{i-1} + 3 s_{i-1}]`. With
`gamma_i = sigma_{i-1} / sigma_i`, standardization under level `i` gives
`m_i / s_i = sqrt(2D)` and

```text
C_i = P_{r ~ N(m_i, s_i^2)}(r in I_{i-1})
    = Phi(sqrt(2D)(gamma_i - 1) + 3 gamma_i)
      - Phi(sqrt(2D)(gamma_i - 1) - 3 gamma_i).
```

The overlap depends on the adjacent pair only through `gamma_i`. Making the overlap constant across
the ladder therefore sets one common ratio `gamma`, so

```text
sigma_i = sigma_1 gamma^{-(i-1)}
log sigma_i = log sigma_1 - (i-1) log gamma.
```

That is a geometric progression, or equal spacing in `log(sigma)`.

To choose the count, solve the overlap equation for a moderate target such as `C ~= 0.5`, then use

```text
L = 1 + (log sigma_1 - log sigma_L) / log gamma.
```

## Endpoints

The top level controls diversity. For a Gaussian mixture centered at data points `x^(i)`, with
responsibilities `r^(j)(x)`,

```text
E_{p^(i)}[r^(j)(x)] <= 0.5 * exp(-||x^(i) - x^(j)||^2 / (8 sigma_1^2)).
```

If `sigma_1` is much smaller than inter-point distances, other modes are exponentially invisible to
the score. Setting `sigma_1` on the order of the maximum pairwise data distance lets distant modes
communicate. The bottom level `sigma_L` is kept small so the last perturbed distribution is close to
the clean data; an optional Tweedie denoising step `x + sigma_L^2 s_theta(x, sigma_L)` removes the
remaining small Gaussian blur.

## Langevin Dynamics

For the one-point toy, Langevin at level `i`,

```text
x_{t+1} = x_t + alpha grad log p_{sigma_i}(x_t) + sqrt(2 alpha) z_t
        = (1 - alpha / sigma_i^2) x_t + sqrt(2 alpha) z_t,
```

with `x_0 ~ N(0, sigma_{i-1}^2 I)` has scalar variance recursion

```text
s_T^2 = (1 - alpha/sigma_i^2)^{2T}
        (sigma_{i-1}^2 - 2 alpha / (1 - (1 - alpha/sigma_i^2)^2))
        + 2 alpha / (1 - (1 - alpha/sigma_i^2)^2).
```

Using `alpha = epsilon sigma_i^2 / sigma_L^2` and dividing by `sigma_i^2`,

```text
s_T^2 / sigma_i^2 =
  (1 - epsilon/sigma_L^2)^{2T}
  (gamma^2 - 2 epsilon / (sigma_L^2 - sigma_L^2(1 - epsilon/sigma_L^2)^2))
  + 2 epsilon / (sigma_L^2 - sigma_L^2(1 - epsilon/sigma_L^2)^2).
```

With a geometric ladder, `gamma` is shared by every adjacent pair, so this relative equilibration
ratio is identical at every level and has no explicit `D` dependence.

## Final Schedule

For `L` levels, indexed `i = 1, ..., L`,

```text
sigma_i = exp(log(sigma_1) + ((i - 1) / (L - 1)) * (log(sigma_L) - log(sigma_1)))
        = sigma_1 * (sigma_L / sigma_1)^((i - 1) / (L - 1)).
```

Equal spacing in `log(sigma)` equals constant ratio, which equals constant adjacent shell overlap.

## Code

Canonical geometric branch:

```python
import numpy as np
import torch


def get_sigmas(config):
    if config.model.sigma_dist == "geometric":
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin),
                               np.log(config.model.sigma_end),
                               config.model.num_classes))
        ).float().to(config.device)
    elif config.model.sigma_dist == "uniform":
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin,
                       config.model.sigma_end,
                       config.model.num_classes)
        ).float().to(config.device)
    else:
        raise NotImplementedError("sigma distribution not supported")
    return sigmas
```

Sampler-domain form with `n + 1` grid nodes, a log floor, and an exact terminal pin:

```python
import math
import torch


def get_sigmas(n, t_min, t_max, device="cpu"):
    log_max = math.log(t_max)
    log_min = math.log(max(t_min, 1e-10))
    sigmas = torch.exp(torch.linspace(log_max, log_min, n + 1))
    sigmas[-1] = t_min
    return sigmas.to(device)
```
