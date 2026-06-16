# RAIN (Recursive Anchored IteratioN), distilled

RAIN finds an `epsilon`-stationary point — a point with small gradient operator norm
`||F(z)|| <= epsilon`, `F(z) = [grad_x f ; -grad_y f]` — of a smooth stochastic minimax problem
`min_x max_y f(x,y) = E_xi[f(x,y; xi)]`, using only a noisy first-order oracle, with near-optimal
stochastic first-order oracle (SFO) complexity. Its key move is a *recursive, moving* anchor:
instead of pulling iterates toward the fixed initial point (as anchored/Halpern extragradient does),
it solves a chain of regularized subproblems whose anchor is re-set to the current iterate and whose
regularization strength grows geometrically, so the anchors converge to the saddle and the
statistical noise term hits its floor.

## Problem it solves

Smooth convex-concave (or strongly-convex-strongly-concave) minimax optimization under an unbiased
oracle with bounded variance `sigma^2`, where the target is the gradient operator norm (not the
duality gap, which is ill-defined for unbounded problems like `f = x^T y`). The aim is a complexity
in which the statistical part (`sigma^2`) and the optimization part (`L`, condition number `kappa`,
initial distance `D = ||z_0 - z*||`) appear *additively*.

## Key idea

- **Anchoring lemma.** With `g(z) = f(z) + (lambda/2)||x - x_0||^2 - (lambda/2)||y - y_0||^2` and
  operator `G(z) = F(z) + lambda(z - z_0)`, for any `z-tilde`:
  `||F(z-tilde)|| <= 2||G(z-tilde)|| + lambda||z_0 - z*||`. Adding the anchor makes `G`
  `lambda`-strongly monotone but pays a bias `lambda||z_0 - z*||`. A fixed anchor at `z_0` therefore
  forces `lambda = Theta(epsilon/D)` (tiny), leaving the noise floor untouched.
- **Non-expansiveness lemma.** If `z*` solves `F(z) = 0` and `w*` solves `G(w) = 0`, then
  `||w* - z_0|| <= ||z* - z_0||` and `||w* - z*|| <= ||z* - z_0||`. The anchored solution lands
  *closer* to `z*` than the anchor was — the warrant for re-anchoring there.
- **Recursive anchoring.** Build `F^(s)(z) = F(z) + sum_{i=1}^s lambda_i (z - z_i)` with anchors
  `z_i` re-set to approximate solutions and geometrically growing strengths. The schedule starts
  with `lambda_0 = lambda gamma` and `lambda_{s+1} = (1+gamma) lambda_s`; in the `gamma = 1`
  analysis the added penalties are `lambda_i = lambda 2^i`, so
  `F^(s)(z) = F(z) + lambda sum_{i=1}^s 2^i (z - z_i)`, for
  `S = floor(log_2(L/lambda))` rounds. In this `gamma = 1` proof each `F^(s)` is at least
  `2^s lambda`-strongly monotone and at most `2L`-Lipschitz, so the subproblems' condition numbers
  shrink toward `O(1)`.
- **Recursive anchoring lemma (`gamma = 1`).** With `z*_s` the exact solution of subproblem `s`,
  `||F(z_S)|| <= 16 lambda sum_{s=1}^S 2^{s-1} ||z*_{s-1} - z_s||`. The final gradient norm is a
  geometrically-weighted sum of per-round subroutine errors; the growing weight is matched by the
  growing strong monotonicity, so each round can be solved just accurately enough.
- **Subproblem solver: Epoch-SEG**, a two-phase stochastic extragradient. Phase 1 (fixed
  `eta = 1/(4L)`, `T = 8L/lambda`, `N` epochs) halves the optimization error per epoch; Phase 2
  (`eta_k = 1/(2^{k-N+3}L)`, `T_k = 2^{k-N+5}L/lambda`, `K` epochs) drives the statistical floor down:
  `E||z_{N+K} - z*||^2 <= 2^{-(N+2K)} E||z_0 - z*||^2 + 8 sigma^2/(2^K lambda L)`, costing
  `<= 16 kappa N + 2^{K+6} kappa` SFO calls. In the recursive calls the smoothness parameter is
  `2L`, so the statistical term used in the induction is `4 sigma^2/(2^K lambda_s L)`.

## Complexity (SCSC and CC)

With `N_0 = ceil(log_2(512 lambda^2 S^2 D^2/epsilon^2))`, `N_s = 3` for `s >= 1`, and
`K_s = ceil(log_2(2048 lambda_s S^2 sigma^2/(L epsilon^2)))`, RAIN returns `E||F(z_S)|| <= epsilon`
with SFO complexity `Otilde(sigma^2 epsilon^{-2} + kappa)` (SCSC). For general convex-concave `f`,
run RAIN on `g(z) = f(z) + (lambda/2)||x - x_0||^2 - (lambda/2)||y - y_0||^2` with
`lambda = min{epsilon/D, L}`; by the anchoring lemma the output is `3epsilon`-stationary for `f`, and
the SFO complexity is `Otilde(sigma^2 epsilon^{-2} + L D epsilon^{-1})`. Both match the lower bounds
`Otilde(sigma^2 epsilon^{-2} + kappa)` and `Otilde(sigma^2 epsilon^{-2} + L D epsilon^{-1})` up to log
factors, so RAIN is near-optimal — the first near-optimal SFO method for finding near-stationary
points of stochastic convex-concave minimax problems.

## Single-loop algorithm

Specializing the nested algorithm to `N_s = 1`, `K_s = 0` collapses the recursion into one
extragradient step whose regularizer is a geometrically-weighted anchor over the stored trajectory:

```
z_{t+1/2} = z_t - eta ( F(z_t; xi)       + lambda gamma sum_{j=0}^{t-1} (1+gamma)^j (z_t       - z_j) )
z_{t+1}   = z_t - eta ( F(z_{t+1/2}; xi) + lambda gamma sum_{j=0}^{t-1} (1+gamma)^j (z_{t+1/2} - z_j) )
```

The anchor sum is `lambda [ (sum_j w_j) z - sum_j w_j z_j ]` with
`w_j = gamma(1+gamma)^j`. The single-loop implementation stores the practical running object as a
scalar `weight_sum = sum_j w_j` and a vector `weighted_flow_sum = sum_j w_j z_j`; the first step has
an empty buffer, and each new `z_next` is then added with weight
`gamma(1+gamma)^(step_index+1)`. Each new iterate becomes the newest, most-heavily-weighted anchor;
`gamma` is kept small so `(1+gamma)^t` never overflows while the order of the method is unchanged.

## Code (single-loop, NumPy)

```python
import numpy as np
from typing import Any


def init_state(problem, initial_z, seed, hyperparameters):
    z0 = as_vector(initial_z, expected_dim=2 * problem.dim)
    return {
        "z": z0,
        "step_index": 0,
        "weight_sum": 0.0,                          # running sum of geometric weights
        "weighted_flow_sum": np.zeros_like(z0),     # running weighted sum of stored iterates
    }


def step(state, oracle, problem, hyperparameters, max_sfo_calls):
    tau = float(hyperparameters["tau"])             # extragradient stepsize eta
    lam = float(hyperparameters["lambda"])          # base regularization strength lambda
    gamma = float(hyperparameters["gamma"])         # geometric growth of anchor weights
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    step_index = int(state.get("step_index", 0))
    weight_sum = float(state.get("weight_sum", 0.0))
    weighted_flow_sum = as_vector(
        state.get("weighted_flow_sum", np.zeros_like(z)), expected_dim=2 * problem.dim
    )

    # extragradient look-ahead with the moving anchor (pull toward weighted past iterates)
    g = oracle.grad(z)
    anchor_z = tau * lam * (weighted_flow_sum - weight_sum * z)
    w = z - tau * g + anchor_z + oracle.noise()

    # extragradient update, anchor re-evaluated at the look-ahead w
    gw = oracle.grad(w)
    anchor_w = tau * lam * (weighted_flow_sum - weight_sum * w)
    z_next = z - tau * gw + anchor_w + oracle.noise()

    # new iterate becomes the newest, most heavily weighted anchor
    current_weight = gamma * (1.0 + gamma) ** (step_index + 1)
    next_state = {
        "z": z_next,
        "step_index": step_index + 1,
        "weight_sum": weight_sum + current_weight,
        "weighted_flow_sum": weighted_flow_sum + current_weight * z_next,
    }
    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(next_state, metric_iterate, 2)


def get_hyperparameters(problem_name, sigma):
    if problem_name == "bilinear":
        return {"tau": 0.1, "lambda": 0.1, "gamma": 0.001}
    if problem_name == "delta_nu":
        return {"tau": 1.0, "lambda": 0.01, "gamma": 0.0001}
    raise KeyError(f"Unknown problem: {problem_name}")
```

This single-loop form uses two oracle evaluations per iteration, additive Gaussian update noise, and
the running-anchor regularizer; the `tau`/`lambda`/`gamma` values are the tuned per-problem constants.
