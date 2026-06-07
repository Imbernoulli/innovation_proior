# Graduated Non-Convexity (GNC)

## Problem

Estimate a model from measurements when a large fraction are *outliers* (gross errors), or
reconstruct a piecewise-smooth field across *discontinuities*. Least squares
`min_x Σ_i r_i(x)²` is non-robust: the influence of a residual, `ψ = ρ' = 2r`, grows without
bound, so a few far-off points dominate the fit. Robustness requires a *redescending* cost
`ρ` whose influence `ψ → 0` for large residuals (quadratic near zero, flat far out), e.g. the
truncated quadratic or Geman–McClure. But every such `ρ` is **non-convex**: descent lands in a
local minimum chosen by the initialization, not the correct global one.

## Key idea

Do not minimize the non-convex robust cost directly. Build a one-parameter family of surrogate
costs `ρ_μ` such that for one extreme of the control `μ` the problem starts from an easy
convexified member, and as `μ` moves to the other extreme `ρ_μ` recovers the true non-convex `ρ`.
Minimize the easy member with no hand-built initial guess, then
**slowly deform `μ` toward the true cost, re-minimizing from the previous solution each time**,
tracking the minimizer along the continuation path from the easy landscape into the hard one. There
is no global-optimality certificate for the non-convex cost; the bet is that small deformations keep
the tracked minimizer in the basin descended from the easy solution, so no initial guess is needed.
This is a deterministic continuation — graduating the non-convexity in — far cheaper than stochastic
annealing.

Each inner minimization is made cheap by the **Black–Rangarajan duality**: introduce explicit
per-measurement weights `w_i ∈ [0,1]` ("trust"), so that
`min_x Σ_i ρ(r_i) ≡ min_{x, w_i∈[0,1]} Σ_i [ w_i r_i² + Φ_ρ(w_i) ]`.
The penalty `Φ_ρ` is the conjugate of `φ(r²) ≜ ρ(r)`, valid whenever `φ` is concave with
`φ'(0)=1`, `φ'(∞)=0`. The joint problem is solved by alternating a **variable update** (weighted
least squares — global, closed form) and a **weight update** (closed form per point). The
surrogate's control parameter survives into `Φ_{ρ_μ}`, so the same continuation runs on the
weights, hardening them from "trust everyone" toward a clean inlier/outlier decision.

Two concrete costs (`c̄²` = inlier noise bound; `r̂_i²` = current squared residual):

- **Geman–McClure.** `ρ(r) = c̄² r²/(c̄² + r²)`, surrogate `ρ_μ(r) = μ c̄² r²/(μ c̄² + r²)`.
  Convex (quadratic) as `μ→∞`, true cost at `μ=1`. Penalty `Φ_{ρ_μ}(w)=μ c̄²(√w − 1)²`;
  weight `w_i = ( μ c̄² / (r̂_i² + μ c̄²) )²`. Continuation: `μ` **decreases** toward 1.

- **Truncated least squares.** `ρ(r) = r²` if `r² ≤ c̄²`, else `c̄²`. Surrogate is a quadratic
  bowl, a concave bridge `2c̄|r|√(μ(μ+1)) − μ(c̄² + r²)` on `r² ∈ [μ/(μ+1) c̄², (μ+1)/μ c̄²]`,
  and a flat shoulder; the bridge curvature `ρ_μ'' = −2μ` tends to zero as `μ→0`, giving the
  convexified start, and the surrogate recovers the true cost
  as `μ→∞`. Penalty `Φ_{ρ_μ}(w) = μ(1−w)/(μ+w) c̄²`; weight is thresholded:
  `w_i = 0` if `r̂_i² ≥ (μ+1)/μ c̄²`, `w_i = 1` if `r̂_i² ≤ μ/(μ+1) c̄²`, else
  `w_i = √(c̄² μ(μ+1)/r̂_i²) − μ`. Continuation: `μ` **increases**.

## Algorithm

1. Initialize weights `w_i = 1`; solve the unweighted fit; compute residuals.
2. Set the control to its convexifying value over the data: GM `μ = 2 r²_max/c̄²`;
   TLS `μ = c̄²/(2 r²_max − c̄²)`.
3. Repeat (outer continuation):
   a. **Weight update:** set each `w_i` by the closed form above from the latest `r̂_i²`.
   b. **Variable update:** `x ← argmin_x Σ_i w_i r_i(x)²` (weighted least squares; for nonlinear
      models, any global solver for the outlier-free problem), then recompute residuals and cost.
   c. Step the control toward the true cost: GM `μ ← μ/1.4` (stop at `μ<1`);
      TLS `μ ← 1.4 μ` (stop when `Σ_i w_i r̂_i²` converges or all weights are binary).
4. Return `x`; inliers are the measurements with `w_i ≈ 1`.

## Code

```python
import numpy as np


def weighted_least_squares(A, y, w):
    # Variable update: argmin_x sum_i w_i (A_i x - y_i)^2, closed form.
    W = np.sqrt(w)[:, None]
    Aw, yw = A * W, y * np.sqrt(w)
    return np.linalg.solve(Aw.T @ Aw, Aw.T @ yw)


def robust_weight_update(r2, mu, barc2):
    """Closed-form TLS weight update for squared residuals."""
    th_out = (mu + 1.0) / mu * barc2
    th_in = mu / (mu + 1.0) * barc2
    w = np.empty_like(r2, dtype=float)
    out = r2 >= th_out
    inn = r2 <= th_in
    mid = ~(out | inn)
    w[out] = 0.0
    w[inn] = 1.0
    w[mid] = np.sqrt(barc2 * mu * (mu + 1.0) / r2[mid]) - mu
    return w


def initial_control_value(r2, barc2):
    return barc2 / (2.0 * float(np.max(r2)) - barc2)


def are_binary_weights(w, threshold=1e-12):
    return bool(np.all((w <= threshold) | (np.abs(1.0 - w) <= threshold)))


def continuation_robust_fit(A, y, barc2, factor=1.4, max_iter=1000, cost_threshold=0.0):
    """GNC-TLS port of the weighted-solver implementation."""
    w = np.ones(A.shape[0])
    x = weighted_least_squares(A, y, w)
    r2 = (A @ x - y) ** 2
    mu = initial_control_value(r2, barc2)
    prev_cost = 0.0

    for _ in range(max_iter):
        w = robust_weight_update(r2, mu, barc2)
        x = weighted_least_squares(A, y, w)
        r2 = (A @ x - y) ** 2
        cost = float(w @ r2)
        cost_diff = abs(cost - prev_cost)
        prev_cost = cost
        if cost_diff < cost_threshold or are_binary_weights(w):
            break
        mu *= factor

    inliers = np.flatnonzero(w > 1.0 - np.finfo(float).eps)
    return x, w, inliers
```

The code needs from each problem only a weighted global solver for the outlier-free model and a
noise-bound threshold `c̄²`; the continuation and thresholded weight update are otherwise unchanged.
