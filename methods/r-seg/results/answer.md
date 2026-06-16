# R-SEG: Regularized (anchored) Stochastic ExtraGradient

## Problem

Smooth convex-concave saddle / monotone variational inequality solved through a noisy oracle:
`min_x max_y f(x,y)`, with saddle gradient operator `F(z) = (grad_x f, -grad_y f)`, `z=(x,y)`. The oracle returns
unbiased noisy queries `F(z;xi) = F(z) + xi`, `E[xi]=0`, `E||xi||^2 <= sigma^2`. Goal: a **near-stationary
point**, `|| F(z) || <= eps` (the robust metric — the duality gap is ill-defined off the saddle, e.g. infinite for
`f=xy`). Two obstacles: simultaneous gradient descent-ascent diverges on convex-concave problems (rotational
operator), and on a merely monotone `F` there is no contraction to drive the last-iterate gradient norm down.

## Key idea

Combine two ingredients.

1. **Extragradient lookahead (Korpelevich 1976).** Evaluate the operator at a tentative point and step from the
   original point with that gradient, so the step picks up the toward-the-saddle component that simultaneous
   descent-ascent misses. This is an explicit approximation of the proximal-point step and converges for monotone
   + Lipschitz operators.

2. **Tikhonov / regularization trick (Nesterov 2012, "How to make the gradients small").** Convex-concave `F` is
   only monotone, not strongly monotone, so add curvature by hand: regularize toward a fixed **anchor** `a = z_0`
   (the initial point),
   ```
   G(z) = F(z) + lambda (z - z_0),
   ```
   the gradient operator of `f(x,y) + (lambda/2)||x - x0||^2 - (lambda/2)||y - y0||^2`. `G` is `lambda`-strongly
   monotone for free, so the extragradient method contracts on it. Solving `G` instead of `F` is legitimate
   because of the **anchoring inequality** (below): a small `||G||` gives a small `||F||` up to a bias controlled
   by `lambda`.

R-SEG = stochastic extragradient run on the regularized operator `G`, with the anchor fixed at `z_0`.

## The update

One iteration (step size `tau = eta`, regularization weight `lambda`, anchor `a = z_0`):
```
w      = z - tau F(z) + tau lambda (z_0 - z) + xi_1       # SEG half-step on G = F + lambda(.- z_0)
z_next = z - tau F(w) + tau lambda (z_0 - w) + xi_2       # SEG full step, gradient at the lookahead w
```
Two independent stochastic operator queries per iteration. Choose `lambda = min(eps/D, L)` with `D >= ||z_0 - z*||`
and `tau < 1/(4(L+lambda))`.

## Why it is correct (the two supporting lemmas)

**Non-expansiveness.** For monotone `F`, with `z*` solving `F(z)=0` and `w*` solving `G(w)=0`:
`||w* - z_0|| <= ||z* - z_0||` and `||w* - z*|| <= ||z* - z_0||`.
*Proof.* `lambda||w*-z*||^2 <= G(z*)^T(z*-w*)` (strong monotonicity of `G`, `G(w*)=0`)
`= lambda(z*-z_0)^T(z*-w*)` (since `F(z*)=0`)
`= (lambda/2)(||w*-z*||^2 + ||z*-z_0||^2 - ||w*-z_0||^2)`. Rearranging gives
`||w*-z*||^2 + ||w*-z_0||^2 <= ||z*-z_0||^2`, hence both bounds. QED.

**Anchoring inequality.** For any `z~`, `|| F(z~) || <= 2 || G(z~) || + lambda || z_0 - z* ||`.
*Proof.* `||F(z~)|| = ||G(z~) - lambda(z~-z_0)|| <= ||G(z~)|| + lambda||z~ - z_0||`
`<= ||G(z~)|| + lambda||z~ - w*|| + lambda||w* - z_0||`
`<= ||G(z~)|| + ||G(z~)|| + lambda||z* - z_0||`,
using `lambda||z~-w*|| <= ||G(z~)-G(w*)|| = ||G(z~)||` (strong monotonicity) and `||w*-z_0|| <= ||z*-z_0||`
(non-expansiveness). QED.

So with `||z_0-z*|| <= D` and `lambda ~ eps/D`, the bias `lambda D ~ eps`; driving `||G(z~)|| <= eps` (which the
strongly-monotone SEG can do) yields `||F(z~)|| = O(eps)`. `lambda` trades the regularization bias `lambda D`
against the conditioning `L/lambda` and the noise floor.

## SEG descent lemma (the contraction on G)

For `G` `lambda`-strongly monotone and `L_G`-Lipschitz, step `0 < eta < 1/(4 L_G)`, unbiased oracle with variance
`sigma^2`, the half-iterate satisfies
```
lambda E|| z_{t+1/2} - w* ||^2 <= (1/eta) E[ ||z_t - w*||^2 - ||z_{t+1} - w*||^2 ] + 16 eta sigma^2.
```
Telescoping over `t=0..T-1` and averaging:
`E|| zbar - w* ||^2 <= ||z_0 - w*||^2 / (lambda eta T) + 16 eta sigma^2 / lambda`. The optimization error decays
`O(1/T)`; the fixed-step statistical term `16 eta sigma^2 / lambda` is the noise floor (shrink `eta` / epoch the
step to reduce it).
*Derivation sketch.* Decompose `2 E[G(z_{t+1/2})^T(z_{t+1/2}-w*)]` into (i) second-query noise vs. distance,
which is zero for an unbiased oracle because `z_{t+1/2}` is fixed before the second query is drawn; (ii) full-step
gradient vs. new distance (`G(z_{t+1/2};xi)=(z_t-z_{t+1})/eta`, polarization -> the telescoping pair and
`-(1/eta)||z_t-z_{t+1}||^2`); (iii) query-difference vs. half-to-full move (Young + `3`-term split + `L_G`-
Lipschitz; `eta<=1/(4L_G)` gives `6 eta L_G^2 <= 1/(2 eta)`, so the Lipschitz square is absorbed and the two
variance terms leave `12 eta sigma^2 <= 16 eta sigma^2`); (iv) half-step gradient vs. the same move
(`G(z_t;xi)=(z_t-z_{1/2})/eta`, polarization; its `+(1/eta)||z_t-z_{t+1}||^2` cancels (ii)'s negative). Strong
monotonicity supplies the left side, and the remaining distance term has the decreasing sign
`||z_t-w*||^2 - ||z_{t+1}-w*||^2`.

## Code

Grounded in the reference convex-concave saddle harness (oracle exposes `grad(z) = F(z)` and `noise() ~ N(0, sigma^2 I)`).

```python
from __future__ import annotations
from typing import Any
import numpy as np
from fixed_benchmark import (
    ProblemSpec, StepOutput, StochasticOracle,
    as_vector, make_step_output, run_cli,
)


def init_state(problem, initial_z, seed, hyperparameters):
    z0 = as_vector(initial_z, expected_dim=2 * problem.dim)
    return {"z": z0, "anchor_z": z0.copy(), "step_index": 0}   # anchor a = z_0, fixed forever


def step(state, oracle, problem, hyperparameters, max_sfo_calls):
    tau = float(hyperparameters["tau"])       # extragradient step eta, < 1/(4(L+lambda))
    lam = float(hyperparameters["lambda"])    # regularization weight, ~ eps / D
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    anchor_z = as_vector(state["anchor_z"], expected_dim=2 * problem.dim)
    step_index = int(state.get("step_index", 0))

    # SEG half-step on G(z) = F(z) + lambda (z - anchor):
    g = oracle.grad(z)
    w = z - tau * g + tau * lam * (anchor_z - z) + oracle.noise()
    # SEG full step on G, gradient evaluated at the lookahead w:
    gw = oracle.grad(w)
    z_next = z - tau * gw + tau * lam * (anchor_z - w) + oracle.noise()

    metric_iterate = z_next if problem.name == "bilinear" else z   # ||F||=||z|| for bilinear
    return make_step_output(
        {"z": z_next, "anchor_z": anchor_z, "step_index": step_index + 1},
        metric_iterate,
        2,   # two stochastic operator queries per iteration
    )


def get_hyperparameters(problem_name, sigma):
    if problem_name == "bilinear":
        return {"tau": 0.1, "lambda": 0.1}
    if problem_name == "delta_nu":
        return {"tau": 1.0, "lambda": 0.01}
    raise KeyError(f"Unknown problem: {problem_name}")


if __name__ == "__main__":
    run_cli(init_state=init_state, step=step, get_hyperparameters=get_hyperparameters)
```
