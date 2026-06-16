## Research question

We want to solve the smooth stochastic minimax problem

```
min_x max_y f(x,y),   f(x,y) = E_xi[ f(x,y; xi) ],
```

where `f` is `L`-smooth, accessed only through a stochastic first-order oracle (SFO) `F(z; xi)`
that is unbiased, `E[F(z;xi)] = F(z)`, with bounded variance `E||F(z;xi) - F(z)||^2 <= sigma^2`.
Here `z = (x,y)` and the *gradient operator* (saddle operator) is

```
F(x,y) = [ grad_x f(x,y) ; -grad_y f(x,y) ].
```

The precise goal is to find an `epsilon`-stationary point — a point with small gradient operator
norm, `||F(z)|| = ||grad f(z)|| <= epsilon` — using as few SFO calls as possible. The gradient
norm, not the duality gap, is the target on purpose: the duality gap
`max_{y'} f(x,y') - min_{x'} f(x',y)` is not always finite or even well-defined (for the bilinear
`f(x,y) = x^T y` it is infinite at every point except the saddle `(0,0)`), whereas the gradient
operator norm is always defined for a differentiable objective, is easy to measure in practice,
and is meaningful even without convex-concavity. What would a satisfying answer have to achieve?
A near-optimal SFO complexity in which the statistical part (driven by `sigma^2`) and the
optimization part (driven by `L`, the condition number, and the initial distance
`D = ||z_0 - z*||`) appear *additively* rather than multiplied together. The state of the art for
making the *gradient* small under noise multiplies them, and closing that gap is the problem.

## Background

**The saddle operator and monotonicity.** For a smooth convex-concave (CC) `f`, the gradient
operator `F` is monotone (Rockafellar 1970): `(F(z) - F(z'))^T (z - z') >= 0` for all `z, z'`.
If `f` is `lambda`-strongly-convex-`lambda`-strongly-concave (SCSC), then `F` is `lambda`-strongly
monotone: `(F(z) - F(z'))^T (z - z') >= lambda ||z - z'||^2`. Smoothness means `F` is
`L`-Lipschitz. These operator-level facts — not convexity of a scalar objective — are the natural
language for minimax, because there is no single scalar whose value the two players agree to
descend.

**Why naive gradient methods fail here.** Simultaneous stochastic gradient descent-ascent diverges
even on the trivial bilinear `f(x,y) = x^T y`: the dynamics rotate around the saddle and spiral
outward (the well-documented cycling behaviour of first-order minimax methods; Pethick et al.
2022). The standard cure is the **extragradient** method (Korpelevich 1976): take a look-ahead
half step, then step from the original point using the gradient at the look-ahead point. This
look-ahead damps the rotation and makes monotone problems converge.

**Two suboptimality measures, two literatures.** For *duality gap* on bounded domains,
extragradient / mirror-prox attain an ergodic `O(1/k)` rate, which is order-optimal, and optimal
*stochastic* algorithms in duality gap are known. For the *gradient norm*, the picture was
different and more recent: it was shown that combining extragradient steps with an *anchoring*
mechanism — a Halpern-style pull of every iterate back toward a fixed reference point with an
iteration-dependent weight — gives the optimal deterministic last-iterate rate `O(L^2 D^2 / k^2)`
on the squared gradient norm. Anchoring (Halpern 1967; brought into minimax by Diakonikolas 2020
and made accelerated by the anchored-extragradient line) is the load-bearing primitive of the
gradient-norm story: a contraction toward a chosen point that, scaled by a vanishing
`beta_k = 1/(k+2)`, drives the last iterate's gradient to zero.

**A diagnostic from the convex (single-player) world.** For convex *minimization* there is a clean
account of how to make the gradient small stochastically (Allen-Zhu 2018). The relevant
observation: regularizing `f(x)` by `(sigma/2)||x - x_0||^2` and solving the strongly convex
surrogate makes the gradient small, but the surrogate's minimizer is displaced from the true one
by `O(sigma)`, so `sigma` must be kept on the order of `epsilon`, and a small `sigma` makes the
surrogate poorly conditioned and the stochastic solve expensive — capping the achievable rate. The
escape observed there: if the regularization center `x_0` were already close to the optimum, a
*larger* `sigma` would still keep the displacement small, and a larger `sigma` means a
better-conditioned, cheaper surrogate; a chain of progressively warmer centers, with the center
re-set to the current approximate minimizer and the regularization strength doubled each round,
breaks the barrier. That account is built on the inequality `min_x f(x) <= f(x)`, an artifact of
single-objective convexity. The same scalar-value argument has no counterpart for a saddle problem,
where neither `min-max f <= f` nor `min-max f >= f` holds.

**The noise floor.** A complexity decomposition (Foster et al. 2019) splits the lower bound for
finding a small-gradient point into a *statistical* part forced by oracle noise and an
*optimization* part forced by smoothness/conditioning. For these minimax problems the statistical
part is `Omega(sigma^2 epsilon^{-2})` and the optimization part is `Omega(kappa)` (SCSC) or
`Omega(L D epsilon^{-1})` (CC). An ideal algorithm would match `Otilde(sigma^2 epsilon^{-2} + kappa)`
and `Otilde(sigma^2 epsilon^{-2} + L D epsilon^{-1})` respectively — additive, not multiplicative.

## Baselines

**Stochastic extragradient, SEG (Korpelevich 1976; stochastic analyses e.g. Mishchenko et al.
2020).** One iteration:

```
z_{t+1/2} = z_t - eta * F(z_t; xi_i)
z_{t+1}   = z_t - eta * F(z_{t+1/2}; xi_j)
```

For SCSC `f` with `eta < 1/(4L)`, one step satisfies the telescoping bound
`lambda E||z_{t+1/2} - z*||^2 <= (1/eta) E[ ||z_t - z*||^2 - ||z_{t+1} - z*||^2 ] + 16 eta sigma^2`,
and averaging gives `E||z_bar - z*||^2 <= ||z_0 - z*||^2/(lambda eta T) + 16 eta sigma^2 / lambda`.
**Limitation:** with a fixed stepsize SEG converges only to a `sigma^2`-sized ball around `z*` and
then stalls; to shrink the ball one must shrink `eta`, which slows the optimization-error decay.
For the gradient norm in the general CC case its complexity is `O(sigma^2 L^2 epsilon^{-4} +
L^2 D^2 epsilon^{-2})` — the statistical term carries an `L^2` and an `epsilon^{-4}`, far above the
`sigma^2 epsilon^{-2}` floor.

**Regularized SEG, R-SEG (regularization trick, Nesterov 2012).** Run SEG not on `f` but on the
regularized surrogate `g(x,y) = f(x,y) + (lambda/2)||x - x_0||^2 - (lambda/2)||y - y_0||^2` for a
small `lambda`, i.e. add a fixed pull `lambda(z_0 - z)` toward the initial point at both half
steps. The surrogate is strongly monotone, so SEG on it is better behaved. **Limitation:** the
surrogate's saddle is displaced from the true one by `O(lambda ||z_0 - z*||)`, so `lambda` must
stay `O(epsilon/D)`; that small `lambda` leaves the surrogate barely strongly monotone and the
stochastic solve nearly as noise-limited as plain SEG. The pull is fixed in both strength and
center for the whole run.

**Stochastic Extra-Anchored Gradient, SEAG (Lee & Kim 2021, the stochastic anchored extragradient
line; deterministic predecessor EAG, Yoon & Ryu 2021).** Extragradient with a Halpern anchor to
the initial point, the anchor weight decaying over time:

```
z_{t+1/2} = z_t - (1 - 1/(t+1)) * eta * F(z_t; xi_i) + (1/(t+1)) * (z_0 - z_t)
z_{t+1}   = z_t -                  eta * F(z_{t+1/2}; xi_j) + (1/(t+1)) * (z_0 - z_t)
```

In the deterministic case this anchoring is exactly what buys the optimal `O(L^2 D^2 / k^2)`
squared-gradient-norm rate, i.e. `O(L epsilon^{-1})` first-order complexity. **Limitation:** the
stochastic guarantee remains tied to the fixed initial anchor and still has complexity
`O(sigma^2 L^2 epsilon^{-4} + L D epsilon^{-1})` — the same bad `sigma^2 L^2 epsilon^{-4}`
statistical term as SEG. (It is also observed that SEAG can diverge under additive noise when the
per-step noise condition `sigma_k^2 <= epsilon/(k+1)` it relies on is violated.)

**Primal-dual hybrid gradient, PDHG (Zhao 2022).** An accelerated stochastic primal-dual scheme
that is optimal in *duality gap*; for SCSC problems it reaches `Otilde(kappa sigma^2 epsilon^{-2} +
kappa)` for the gradient norm. **Limitation:** the statistical term carries an extra factor of the
condition number `kappa` beyond the `sigma^2 epsilon^{-2}` floor, and it is tailored to the gap
metric rather than the gradient norm.

Across these, the recurring shape of the gap is the same: the methods that make the gradient small
(anchored extragradient) pay an `epsilon^{-4}` statistical price under noise, and the regularized
methods that exploit strong monotonicity are held back by a small fixed regularization strength.

## Evaluation settings

The natural yardsticks for a small-gradient minimax method:

- **Scalar / low-dimensional bilinear problem** `f(x,y) = x^T y`. The canonical stress test: its
  only saddle is `(0,0)`, its duality gap is infinite elsewhere, simultaneous gradient
  descent-ascent diverges, and even careful first-order methods converge slowly because of cycling.
  A standard instance uses a fixed start `z_0 = [10, 10]^T`, a fixed step `tau`, and additive
  Gaussian update noise of scale `sigma`.
- **The "hard" `(delta, nu)` convex-concave problem** (worst-case construction in the style of
  Yoon & Ryu 2021): `f_{delta,nu}(x,y) = (1-delta) g_nu(x) + delta x^T y - (1-delta) g_nu(y)`, with
  `g_nu` a Huber-type function (quadratic for `|u| < nu`, linear beyond). Standard values
  `nu = 5e-5`, `delta = 1e-2`, dimension `d = 100`, a Gaussian-random start `z_0`, fixed step
  `tau = 1`, additive Gaussian noise `sigma`.
- **Stochastic first-order oracle** that returns the exact gradient operator corrupted by additive
  Gaussian noise, `F(z; xi) = F(z) + xi`, `xi ~ N(0, sigma^2 I)`, drawn fresh at each operator
  evaluation; each iteration uses two evaluations (the extragradient look-ahead and the update).
- **Metric:** the gradient operator norm `||F(z)||` plotted against the number of iterations / SFO
  calls, on a log scale, for several noise levels `sigma`. Lower is better; the headline number is
  the final gradient norm. Step size, regularization strength, and any scalar constants used by the
  update rule are selected by a small grid search.

## Code framework

A single-loop stochastic minimax solver plugs into a fixed benchmark harness. The harness owns the
problem (the deterministic gradient operator and the additive-Gaussian noise oracle), the fixed
initial point, the per-problem iteration count, and the gradient-norm metric. What is *not* settled
is the update rule and the state it keeps — that is exactly what is to be designed. The substrate
below is only the generic extragradient machinery that already exists: a state dict that must
preserve the starting point, a `step` that consumes two oracle evaluations per iteration, and a
per-problem hyperparameter map.

```python
import numpy as np
from typing import Any


def init_state(problem, initial_z, seed, hyperparameters):
    """Return the per-run state. Must preserve the starting point in state['z'].
    Any extra buffers the update rule needs are created here."""
    z0 = as_vector(initial_z, expected_dim=2 * problem.dim)
    state = {"z": z0, "step_index": 0}
    # TODO: any additional state the update rule we design will maintain.
    return state


def step(state, oracle, problem, hyperparameters, max_sfo_calls):
    """One iteration. The oracle exposes a deterministic gradient operator
    oracle.grad(z) and a fresh additive-Gaussian draw oracle.noise(); an
    extragradient iteration uses two gradient evaluations."""
    eta = float(hyperparameters["tau"])
    z = as_vector(state["z"], expected_dim=2 * problem.dim)

    # one extragradient iteration with whatever update rule we design
    g = oracle.grad(z)
    # TODO: the update rule we will design — compute the look-ahead point w,
    #       then the next iterate z_next, from the gradients and (optionally)
    #       any state we keep across iterations, plus the oracle noise.
    w = z - eta * g + oracle.noise()                     # placeholder look-ahead
    gw = oracle.grad(w)
    z_next = z - eta * gw + oracle.noise()               # placeholder update
    next_state = {"z": z_next, "step_index": int(state["step_index"]) + 1}

    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(next_state, metric_iterate, 2)  # 2 SFO calls


def get_hyperparameters(problem_name, sigma):
    """Per-problem constants for the update rule."""
    # TODO: the constants the update rule needs, per problem.
    raise NotImplementedError
```

The harness draws the fixed start, calls `init_state` once, then calls `step` for the fixed number
of iterations, recording `||F(metric_iterate)||` each time. The single empty slot is the update
rule together with whatever per-iteration state it chooses to carry.
