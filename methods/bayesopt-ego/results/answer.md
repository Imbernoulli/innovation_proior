# Efficient Global Optimization (EGO): GP/kriging surrogate + Expected Improvement

## Problem

Minimize an **expensive, deterministic, gradient-free black-box** function `y(x)` over a box of
design variables, using **as few evaluations as possible** (budget of tens, because one call is a
multi-hour simulation/experiment), while coping with possible **multimodality** and ideally
knowing **when to stop**. Grid/random/local/true-function global optimizers all need far more
evaluations than affordable and either ignore the data they collect or fall into a local minimum.

## Key idea

Spend cheap reasoning to save expensive calls: fit a **cheap probabilistic surrogate** to the
evaluations so far, then let the surrogate choose the next point.

1. **Surrogate (kriging / DACE).** Model `y(x) = μ + ε(x)` with `ε` a zero-mean Gaussian random
   field, `Corr(ε(x_i), ε(x_j)) = exp(−Σ_h θ_h |x_{ih} − x_{jh}|^{p_h})` (θ_h = per-variable
   activity, p_h ∈ [1,2] = smoothness). Fit θ, p by maximum likelihood (μ̂, σ̂² close in closed
   form). This yields, at any `x`, both a prediction `ŷ(x)` that **interpolates** the data and an
   honest local error bar `s(x)` — zero at sampled points, growing to σ far from them. A point
   prediction alone (ordinary regression/splines) can't say where it's ignorant, so it can't drive
   global search.

2. **Acquisition (Expected Improvement).** Treat the unknown value as `Y(x) ~ Normal(ŷ, s²)` and
   define improvement `I = max(f_min − Y, 0)` with `f_min` the best value seen. Maximizing the
   *probability* of improvement (Kushner 1964) ignores magnitude and hugs the incumbent; instead
   maximize the **expected** improvement, `E[I] = E[max(f_min − Y, 0)]`, which weights each
   possible gain by how big *and* how likely it is. Closed form (let `z = (f_min − ŷ)/s`):

   **EI(x) = (f_min − ŷ) Φ(z) + s φ(z)**

   The first term is **exploitation** (large when ŷ is below the incumbent), the second is
   **exploration** (large when s is large). The trade-off is intrinsic — no hand-tuned knob.
   EI = 0 at sampled points, is positive between, and is multimodal, so iterating it searches
   **globally**. It is monotone — `∂EI/∂ŷ = −Φ(z) < 0`, `∂EI/∂s = φ(z) > 0` — so branch-and-bound
   can upper-bound EI over a box by lower-bounding `ŷ` and upper-bounding `s`. The `s²` bound comes
   from a convex relaxation of the kriging MSE; the fast `ŷ` bound uses `p_h = 2`, writes
   `r_i = exp(-z_i)`, and underestimates each `c_i exp(-z_i)` by a tangent if `c_i >= 0` or a chord
   if `c_i < 0`. EI's own magnitude is a **stopping rule**.

## Algorithm

1. **Initial design.** Latin hypercube over the box (space-filling, good low-dimensional
   projections, no clustering); evaluate `y` on it.
2. **Fit** the kriging surrogate by maximum likelihood; diagnose with cross-validated standardized
   residuals (< ~3); if poor, model `log y` or `−1/y`.
3. **Iterate.** Maximize EI over the box. If `max EI < ~1%` of `|f_min|`, **stop**. Otherwise
   evaluate `y` at the EI-argmax (one expensive call), append it, refit, repeat.
4. **Conditioning.** Solve through an SVD of `R` (zero tiny singular values) / a small nugget so
   `R` stays invertible when the surface is very smooth or points cluster late in the run.

Modern realization: a Gaussian-process regressor with a Matérn kernel (ν ↔ smoothness `p`) and ARD
length scales (↔ activity `θ_h`) plus a tiny numerical nugget, driven by an expected-improvement
acquisition in an ask/fit/maximize/tell loop, matching the structure of skopt's `gp_minimize`.

## Code

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


def latin_hypercube(n_points, bounds, rng):
    bounds = np.asarray(bounds, float)
    k = len(bounds)
    cut = np.linspace(0.0, 1.0, n_points + 1)
    u = rng.uniform(size=(n_points, k))
    pts = cut[:n_points, None] + u * (cut[1] - cut[0])
    for j in range(k):
        pts[:, j] = rng.permutation(pts[:, j])
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + pts * (hi - lo)


class CorrelatedSurrogate:
    """GP/kriging surrogate with ARD length scales and a tiny numerical nugget."""

    def __init__(self, alpha=1e-10):
        self.alpha = alpha
        self.gp = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        kernel = (ConstantKernel(1.0) *
                  Matern(length_scale=np.ones(X.shape[1]), nu=2.5))
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=self.alpha, normalize_y=True,
            n_restarts_optimizer=10)
        self.gp.fit(X, y)
        return self

    def predict(self, X, return_std=True):
        return self.gp.predict(np.atleast_2d(X), return_std=return_std)


def expected_improvement(X, surrogate, f_min, xi=0.0):
    mu, std = surrogate.predict(X, return_std=True)
    mu = np.atleast_1d(mu); std = np.atleast_1d(std)
    ei = np.zeros_like(mu)
    mask = std > 1e-12                               # EI = 0 where s = 0
    improve = f_min - xi - mu[mask]                  # f_min - mu when xi=0
    z = improve / std[mask]                          # z = improve / s
    ei[mask] = improve * norm.cdf(z) + std[mask] * norm.pdf(z)
    #          \_ exploitation                       \_ exploration
    return ei


def maximize_acquisition(acq_fn, bounds, rng, n_restarts=20, n_raw=10000):
    bounds = np.asarray(bounds, float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    raw = lo + rng.uniform(size=(n_raw, len(bounds))) * (hi - lo)
    vals = acq_fn(raw)
    seeds = raw[np.argsort(vals)[-n_restarts:]]
    best_x, best_val = raw[vals.argmax()], float(vals.max())
    for x0 in seeds:
        res = minimize(lambda x: -float(acq_fn(x)[0]),
                       x0, bounds=list(map(tuple, bounds)), method="L-BFGS-B")
        if -res.fun > best_val:
            best_x, best_val = res.x, -res.fun
    return best_x, best_val


def efficient_global_optimization(objective, bounds, n_init=10, max_evals=40,
                                  ei_tol_frac=0.01, xi=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = latin_hypercube(n_init, bounds, rng)
    y = np.array([objective(x) for x in X])          # the only expensive calls
    surrogate = CorrelatedSurrogate()
    for _ in range(max_evals - n_init):
        surrogate.fit(X, y)
        f_min = y.min()
        acq_fn = lambda Xcand: expected_improvement(Xcand, surrogate, f_min, xi)
        x_next, ei = maximize_acquisition(acq_fn, bounds, rng)
        if ei < ei_tol_frac * max(abs(f_min), 1e-12):   # EI-based stopping rule
            break
        y_next = objective(x_next)                   # one expensive call
        X = np.vstack([X, x_next]); y = np.append(y, y_next)
    i = y.argmin()
    return X[i], y[i]
```

The EI algebra follows skopt's `gaussian_ei`
(`improve = y_opt - xi - mu; scaled = improve/std; values = improve*Φ(scaled) + std*φ(scaled)`).
Setting `xi=0` recovers the no-margin EI criterion above; skopt exposes `xi` as an optional margin
and commonly defaults it to `0.01`.
