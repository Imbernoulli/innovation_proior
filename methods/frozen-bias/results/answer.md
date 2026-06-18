# Frozen-bias shallow network for single-index models, distilled

The frozen-bias network is a two-layer ReLU model in which all hidden neurons share a single
inner direction `theta`, the per-neuron biases (and signs) are sampled once at random
initialization and *never trained*, and only the shared direction `theta` and the readout `c`
are optimized by gradient descent. Freezing the biases collapses the high-dimensional,
non-convex direction-recovery problem onto a benign one-dimensional landscape in the correlation
`m = <theta, theta*>`, which lets ordinary gradient descent recover the hidden direction `theta*`
with near-optimal sample complexity *and* learn the unknown link `f_*` non-parametrically at a
rate independent of the ambient dimension.

## Problem it solves

Single-index regression `y = f_*(<theta*, x>) + xi`, `x ~ N(0, I_d)`, `theta* in S^{d-1}`, with
*both* the direction `theta*` and the univariate link `f_*` unknown. Recover `theta*`
(parametric, high-dimensional) and estimate `f_*` (non-parametric, one-dimensional) with one
shallow network trained by gradient descent — no prior knowledge of `f_*`, no bespoke estimator.

## Architecture

```
G(x; c, theta) = (1/sqrt(N)) sum_{i=1}^N c_i * phi(eps_i <theta, x> - b_i),   phi = ReLU,
```

with `theta in S^{d-1}` shared across all neurons, `b_i ~ N(0, tau^2)` (`tau > 1`) and
`eps_i in {+1,-1}` Rademacher, both *frozen* at initialization. Trained parameters: `theta` and
`c`. Objective: Tikhonov-regularized squared error `L(c, theta) = E[(y - G)^2] + lambda ||c||^2`
(empirically, the sample average).

## Key idea

- **Tie the inner weights, freeze the biases.** One direction `theta` does the high-dimensional
  inference; the frozen `{(b_i, eps_i)}` form a fixed one-dimensional random-feature dictionary
  in `u = <theta, x>` that the readout `c` combines to build the link. The biases are the *lazy*
  (kernel) part; the direction is the *rich* (feature-learning) part.
- **Landscape collapse.** Because the biases do not depend on `theta`, the population loss
  depends on `theta` only through `m = <theta, theta*>`:

  ```
  L(c, theta) = 1 + c^T Q_lambda c - 2 <c, sum_{j>=s} alpha_j m^j T_j> + sigma^2,
  ```

  with `f_* = sum_j alpha_j h_j` (Hermite), `T_j = T h_j`, `Q = T T^*`, `Q_lambda = Q + lambda I`.
  Hence `nabla_theta L = -(sum_{j>=s} <T_j, c> j alpha_j m^{j-1}) theta*` is colinear with
  `theta*`: the whole `d`-dimensional flow reduces to a scalar flow in `m`.
- **Benign topology.** The projected loss `Lbar(theta) = min_c L = -<g_m, P_lambda g_m> +
  (1+sigma^2)`, `g_m = sum_{j>=s} alpha_j m^j h_j`, `P_lambda = Sigma(Sigma+lambda I)^{-1}`. In
  the ideal limit (`lambda -> 0`, infinite features) `Lbar = -sum_{j>=s} alpha_j^2 m^{2j} +
  (1+sigma^2)` is strictly decreasing in `|m|`: only critical points are the equator `m=0` and
  the poles `m=+-1`. With finite regularized features this is preserved (see below).
- **Information exponent sets the rate.** With `s = min{j : alpha_j != 0}`, near the equator the
  signal scales like `m^{s-1}`; the random init sits at `m_0 = Theta(1/sqrt(d))`; beating the
  uniform empirical-gradient error `~ sqrt(d/n)` gives recovery at `n = Theta(d^s)`.

## Why the finite-feature landscape stays benign

Critical-point equation (dividing the `dLbar/dm = 0` condition by 2):

```
sum_{j>=s} alpha_j^2 j m^{2j-1} = <(I - P_lambda) g_m, gbar_m>_gamma,   gbar_m = sum_{j>=s} j alpha_j m^{j-1} h_j.
```

- **Left side** `>= alpha_s^2 s |m|^{2s-1}` (leading Hermite term dominates).
- **Right side**: Cauchy-Schwarz + random-feature approximation. With `N >= (C/lambda)
  log(1/(lambda delta))` features (degrees-of-freedom bound, Bach 2017b), `||(I-P_lambda) f||^2
  <= 4 A(f, lambda)` for zero-mean `f`. RKHS approximation error (ReLU Sobolev representation):

  ```
  A(f, lambda) <= C ( tau^{1+beta} ||f''||_4^2 * lambda^beta + lambda C_f^2 ),   beta = (1 - 1/tau^2)/(3 + 1/tau^2).
  ```

  `beta > 0` requires `tau > 1`. Both sides scale like `|m|^{2s-1}`, so for

  ```
  lambda < lambda* = ( 4 sqrt(C tau^{1+beta}) Ktilde C_{f_*}^2 / (alpha_s^2 s) )^{-2/beta}
  ```

  the right side is strictly below the left for all `m != 0` — contradiction, no critical point
  with `0 < |m| < 1`. `N ~ 1/lambda` features suffice; `N` is independent of `d`.

## Recovery guarantee

Uniform gradient concentration `sup ||nabla L_n - nabla L|| = O(r^2 sqrt(D/n))`, `D = max{d, N}`
(at the ReLU kink, controlled by Gaussian anti-concentration of activation flips), transfers the
benign topology to the empirical landscape. Run two-phase gradient flow (Riemannian on `theta`):
phase 1 optimizes only `theta` from a small/sparse readout `c(0)` (`||c(0)||_0 = N_0`, norm
`rho`) until `T_0 = Otilde(d^{s/2-1})`; phase 2 jointly optimizes `c, theta`. Then with constant
probability,

```
1 - |<theta_T, theta*>| = Otilde( lambda^{-4} max{ (d+N)/n, d^4/n^2 } ),
```

so for `lambda = Theta(1)` and `s > 2` the sample complexity is `n = Theta(d^s)`, near-optimal
relative to the `d^{s-1}` lower bound for gradient methods. The two-phase schedule (lazy/rich
relative scaling) avoids the `c`-`theta` entanglement that would otherwise cost `d^{2s}`.

## Fine-tuning (decoupled non-parametric rate)

Re-fit the readout alone by ridge regression on a *fresh* sample (sample splitting breaks the
data-kernel dependence), with a separate small `lambda'`:

```
chat = argmin_c (1/n') sum_i (c^T Phi(<theta_hat, x_i'>) - y_i')^2 + lambda' ||c||^2.
```

Excess risk splits into a `d`-independent non-parametric term plus a direction-error term:

```
E[ ||F_hat - F_*||^2 ] <~ ||f_*''||^{2/(beta+1)} (sigma^2 tau^2 / n')^{beta/(beta+1)} + ||f_*'||^2 (1 - |m|).
```

Large `lambda` in phase 1 (fast recovery), small `lambda'` in the re-fit (low excess risk).

## Non-smooth optimization note

Squared loss on a ReLU net is non-smooth; "gradient flow" is the subgradient inclusion
`zdot in -partial L(z)` (Clarke). Existence + descent hold because the loss is definable in an
o-minimal structure (admits a chain rule), so `dL/dt = -||partial-bar L||^2 <= 0`; the spherical
constraint is handled by projecting the subdifferential (and `<z, zdot> = 0` on the sphere). A
smooth activation removes this and improves the `s in {1,2}` rates to `d^s`, but ReLU is the
practitioner default.

## Direction estimator

```
theta_hat = normalize( sum_j |c_j| w_j ),
```

`w_j` the first-layer rows, `c_j` the readout weights (a single shared direction in the tied
model; this is the standard estimator for an untied implementation).

## Working code

Filling the shallow-network harness: initialize a shared inner direction on the sphere, freeze
the biases, train direction + readout with SGD on squared loss, optional ridge re-fit.

```python
import math
import torch
import torch.nn as nn


class Strategy:
    """Frozen-bias shallow network: biases sampled once at init and frozen; only the
    (shared) first-layer direction and the readout are trained. Freezing the biases
    collapses the d-dimensional landscape to a benign 1-D landscape in m = <theta, theta*>."""

    def __init__(self, config):
        self.config = config

    def init_two_layer(self, net, config):
        with torch.no_grad():
            W = torch.randn_like(net.fc1.weight)
            W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)   # rows on the unit sphere
            net.fc1.weight.copy_(W)
            net.fc1.bias.uniform_(-1.0, 1.0)                      # biases sampled once...
        net.fc1.bias.requires_grad_(False)                       # ...and FROZEN (the key move)

        bound = 1.0 / math.sqrt(config.width)                    # small readout (lazy/rich scale)
        nn.init.uniform_(net.fc2.weight, -bound, bound)
        nn.init.zeros_(net.fc2.bias)

    def make_optimizer(self, net, config):
        params = [p for p in net.parameters() if p.requires_grad]  # frozen biases excluded
        return torch.optim.SGD(
            params, lr=config.base_lr, momentum=config.momentum,
            weight_decay=config.weight_decay,                      # Tikhonov lambda
        )

    def training_step(self, net, optimizer, x, y, step, config):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        preds = net(x)
        loss = torch.mean((preds - y) ** 2)                        # squared-error objective
        loss.backward()
        optimizer.step()
        return loss

    def finalize(self, net, x_train, y_train, config):
        return                                                     # optional ridge re-fit goes here


def build_strategy(config):
    return Strategy(config)
```
