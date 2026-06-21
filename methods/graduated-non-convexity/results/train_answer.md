The problem is to fit a model or reconstruct a surface when some measurements are gross outliers or when the underlying field is only piecewise smooth. Ordinary least squares fails because the squared residual cost makes the influence of a point grow without bound: a single far-away outlier can dominate the entire fit. The fix from robust statistics is to use a redescending cost that is quadratic near zero but flattens out for large residuals, so gross errors are ignored. The same structure appears in discontinuity-preserving reconstruction, where the smoothness penalty must allow jumps rather than blur them. But any cost strong enough to reject outliers is non-convex, so direct descent lands in whichever local basin the initial guess happens to sit in.

The right idea is not to attack the final non-convex cost directly, but to graduate the non-convexity in through a one-parameter family of surrogates. Start from a convexified version of the cost, find its unique minimizer without any special initialization, then slowly deform the cost back toward the true robust one while re-minimizing from the previous solution at each step. If the deformation is slow enough, the tracked minimizer stays in the basin descended from the easy start instead of falling into a spurious local minimum. The method is called Graduated Non-Convexity, or GNC.

GNC works cleanly because of the Black-Rangarajan duality. For a robust cost whose profile as a function of squared residual is concave, one can introduce an explicit trust weight in [0, 1] per measurement and write the robust problem as a joint minimization over the model and the weights. The first term is a weighted squared residual, and the second term is a penalty that depends only on the weight. Minimizing over the weight recovers the original robust cost exactly. This turns each stage into alternating a weighted least-squares solve, which is global and closed-form for linear models, and a closed-form scalar weight update, which is the soft inlier/outlier decision. The continuation parameter survives into the penalty, so the same gradual hardening runs on the weights.

A particularly practical choice is the truncated least squares surrogate. The true cost is quadratic for squared residuals below a threshold cbar2 and flat above it. The surrogate replaces the sharp kink with a smooth concave bridge whose negative curvature is controlled by a parameter mu. When mu is small the bridge is nearly flat and the problem is convexified; as mu grows the bridge collapses and the surrogate converges to the hard truncated cost. The weight update has three regimes: full trust when the residual is small, no trust when it is large, and a smooth interpolation across the bridge in between. Starting mu from a convexifying value based on the largest residual, then increasing it geometrically, tracks the solution into the non-convex regime and terminates once the weights have become binary.

```python
import numpy as np


def weighted_least_squares(A, y, w):
    W = np.sqrt(w)[:, None]
    Aw, yw = A * W, y * np.sqrt(w)
    return np.linalg.solve(Aw.T @ Aw, Aw.T @ yw)


def robust_weight_update(r2, mu, barc2):
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
