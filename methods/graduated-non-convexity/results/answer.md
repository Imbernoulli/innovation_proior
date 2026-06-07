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
costs `ρ_μ` such that for one extreme of the control `μ` the total energy is **convex** (unique
global minimum, reachable from any start), and as `μ` moves to the other extreme `ρ_μ` recovers
the true non-convex `ρ`. Find the global minimum of the convex member with no initial guess, then
**slowly deform `μ` toward the true cost, re-minimizing from the previous solution each time**,
tracking the minimizer from the easy landscape into the global basin of the hard one. This is a
deterministic continuation — graduating the non-convexity in — far cheaper than stochastic
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
  and a flat shoulder; `ρ_μ'' = −2μ → 0` so it is **convex as `μ→0`** and recovers the true cost
  as `μ→∞`. Penalty `Φ_{ρ_μ}(w) = μ(1−w)/(μ+w) c̄²`; weight is thresholded:
  `w_i = 0` if `r̂_i² ≥ (μ+1)/μ c̄²`, `w_i = 1` if `r̂_i² ≤ μ/(μ+1) c̄²`, else
  `w_i = √(c̄² μ(μ+1)/r̂_i²) − μ`. Continuation: `μ` **increases**.

## Algorithm

1. Initialize weights `w_i = 1`; solve the unweighted fit; compute residuals.
2. Set the control to its convexifying value over the data: GM `μ = 2 r²_max/c̄²`;
   TLS `μ = c̄²/(2 r²_max − c̄²)`.
3. Repeat (outer continuation):
   a. **Variable update:** `x ← argmin_x Σ_i w_i r_i(x)²` (weighted least squares; for nonlinear
      models, any global solver for the outlier-free problem).
   b. **Weight update:** set each `w_i` by the closed form above from `r̂_i²`.
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
    x, *_ = np.linalg.lstsq(Aw.T @ Aw, Aw.T @ yw, rcond=None)
    return x


def gnc_tls(A, y, barc2, factor=1.4, max_iter=1000, tol=1e-6):
    """GNC with truncated-least-squares cost. mu runs 0 (convex) -> inf (true)."""
    N = A.shape[0]
    w = np.ones(N)
    x = weighted_least_squares(A, y, w)
    r2 = (A @ x - y) ** 2
    mu = barc2 / (2.0 * r2.max() - barc2)          # convexifying start
    prev = np.inf
    for _ in range(max_iter):
        x = weighted_least_squares(A, y, w)        # variable update (global)
        r2 = (A @ x - y) ** 2
        th1 = (mu + 1) / mu * barc2                # above -> outlier (w=0)
        th2 = mu / (mu + 1) * barc2                # below -> inlier  (w=1)
        w = np.where(r2 >= th1, 0.0,               # closed-form weight update
             np.where(r2 <= th2, 1.0,
                      np.sqrt(barc2 * mu * (mu + 1) / r2) - mu))
        cost = float(w @ r2)
        if abs(cost - prev) < tol or np.all((w < 1e-10) | (w > 1 - 1e-10)):
            break
        prev = cost
        mu *= factor                               # sharpen the non-convexity
    return x, w


def gnc_gm(A, y, barc2, factor=1.4, max_iter=1000, tol=1e-6):
    """GNC with Geman-McClure cost. mu runs large (convex) -> 1 (true)."""
    N = A.shape[0]
    w = np.ones(N)
    x = weighted_least_squares(A, y, w)
    r2 = (A @ x - y) ** 2
    mu = 2.0 * r2.max() / barc2
    prev = np.inf
    for _ in range(max_iter):
        x = weighted_least_squares(A, y, w)
        r2 = (A @ x - y) ** 2
        w = (mu * barc2 / (r2 + mu * barc2)) ** 2  # closed-form GM weight
        cost = float(w @ r2)
        if mu <= 1.0 or abs(cost - prev) < tol:
            break
        prev = cost
        mu /= factor
    return x, w


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, frac = 100, 0.8                             # 80% gross outliers
    a_gt, b_gt = 2.0, 1.0
    t = rng.uniform(-5, 5, N)
    A = np.column_stack([t, np.ones(N)])
    y = a_gt * t + b_gt + rng.normal(0, 0.1, N)
    out = rng.choice(N, int(frac * N), replace=False)
    y[out] += rng.uniform(-50, 50, len(out))

    barc2 = (3 * 0.1) ** 2
    x_ls, *_ = np.linalg.lstsq(A, y, rcond=None)
    x_tls, w = gnc_tls(A, y, barc2)
    x_gm, _ = gnc_gm(A, y, barc2)
    print("ground truth :", [a_gt, b_gt])         # [2.0, 1.0]
    print("plain LS     :", x_ls.round(3))         # ~[1.42, 2.76]  (wrecked)
    print("GNC-TLS      :", x_tls.round(3))        # ~[2.00, 1.00]  (recovered)
    print("GNC-GM       :", x_gm.round(3))         # ~[2.00, 1.02]
```

Under 80% outliers, plain least squares is dragged far off the true line while both GNC variants
recover it and the TLS weights correctly flag the inlier set. The method needs from each problem
only a (weighted) global solver for the outlier-free model and a robust cost with the duality
structure; everything else is the same continuation loop.
