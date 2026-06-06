# Energy-function minimization with exponent annealing for packing equal circles in a square

## Problem

Place `n` equal non-overlapping circles in the unit square to maximize the common radius `r_n`.
Absorbing the wall offset and rescaling turns this into the point-spreading problem: place `n`
points in a unit square to maximize the minimum pairwise distance `d_n = min_{i<j} ‖s_i − s_j‖`.
If the centre square has side `1 − 2r`, rescaling it to side `1` turns a nearest centre spacing
`2r` into `d = 2r/(1 − 2r)`, so `d(1 − 2r) = 2r` and

    r_n = d_n / (2 (d_n + 1)).

The objective `d_n` is a non-smooth `min`-over-pairs, multimodal in the `2n` coordinates, which is
what defeats naive smooth optimizers (direct radius maximization with simplex/BFGS fails for
`n = 14, 15, 17` already).

## Key idea

Relax the hard minimum into a smooth surrogate via the negative-power mean
`min_{i<j} ‖s_i−s_j‖ = lim_{p→−∞} (Σ_{i<j} ‖s_i−s_j‖^p)^{1/p}`. For fixed `p < 0`, the outer
`1/p` power is decreasing, so maximizing this soft minimum is the same as minimizing
`Σ ‖s_i−s_j‖^p`. Put `m = −p > 0` and use inverse squared distances to get the positive-exponent
repulsion energy:

    E = Σ_{1≤i<j≤n} ( λ / d_ij² )^m,        d_ij² = (x_i − x_j)² + (y_i − y_j)².

For large `m` the sum is dominated by the single closest pair, so minimizing `E` maximizes the
minimum distance. Three supports make this practical:

1. **Unconstrained box via the sine map.** Set `x_i = sin(x̃_i)`, `y_i = sin(ỹ_i)` with
   `x̃_i, ỹ_i ∈ R`. Then `|x_i|, |y_i| ≤ 1` automatically (a side-2 box; the unit square up to a
   harmless rescale), so the search over `u = (x̃, ỹ) ∈ R^{2n}` is fully unconstrained. Gradients
   chain through `∂x_i/∂x̃_i = cos(x̃_i)`.

2. **Scale factor λ.** With huge `m`, `(λ/d²)^m` overflows or underflows unless the base sits near
   1. After each stage set `λ = d²_min`, the square of the current shortest distance, so the
   dominant term starts at `1`. `λ` is held fixed inside that stage's objective and gradient; it
   conditions the arithmetic only.

3. **Exponent annealing (homotopy).** Small `m` ⇒ smooth, easy, but a poor proxy (rewards spreading
   all distances). Large `m` ⇒ faithful proxy, but viciously multimodal. Start small (`m` in
   10–100), minimize, then repeatedly double `m`, recompute `λ`, and re-minimize warm-started from
   the previous solution — tracking the minimizer as the objective deforms from soft to sharp. Run
   `m` up to `~10⁶` (occasionally far beyond). Use many random restarts (`≥ 50` per `n`) to cover
   the basins; nudge any point pinned against a wall with empty interior neighbourhood off the wall
   and re-minimize.

## Final algorithm

For each `n`: (a) repeat `≥ 50` times from random starts; (b) within each start, anneal `m` upward,
at each stage setting `λ = d²_min` and minimizing `E` by conjugate gradient / modified Newton with
Goldstein–Armijo line search and the analytic gradient; (c) keep the best configuration; (d)
**refine**: sort all pairwise distances, find the jump separating contacts (≈ `d_n`) from gaps,
treat near-wall points as on-wall, form the resulting nonlinear contact system, and solve it by
Newton–Raphson in arbitrary precision (gaps/overlaps `< 10⁻³³`). In special structured cases this
can be reduced to a univariate polynomial for the diameter; for irregular large cases it gives a
high-precision numerical candidate.

Gradient used by the optimizer with `λ` held fixed (`T_ij = (λ/d_ij²)^m`):

    ∂E/∂x_i = Σ_{j≠i} (−m T_ij / d_ij²) · 2(x_i − x_j),
    ∂E/∂x̃_i = (∂E/∂x_i) · cos(x̃_i),

and likewise for `y`.

## Code

```python
import numpy as np
from scipy.optimize import minimize, least_squares

# x_i = sin(x_tilde_i), y_i = sin(y_tilde_i): box constraint |x|,|y| <= 1 is automatic,
# so the search over u = [x_tilde; y_tilde] in R^{2n} is unconstrained.
def to_box(u, n):
    ut = u.reshape(2, n)
    return np.sin(ut[0]), np.sin(ut[1])              # coords in [-1, 1] (a side-2 box)

def pairwise_sq_dists(x, y):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return dx*dx + dy*dy

def min_pair_distance(u, n):
    x, y = to_box(u, n)
    d2 = pairwise_sq_dists(x, y)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2.min())

# E = sum_{i<j} (lambda / d_ij^2)^m : a 1/r^(2m) repulsion dominated, for large m,
# by the closest pair -> minimizing E maximizes the minimum pairwise distance.
def objective(u, n, sharpness, scale):
    x, y = to_box(u, n)
    d2 = pairwise_sq_dists(x, y)
    iu = np.triu_indices(n, 1)
    return np.sum((scale / d2[iu])**sharpness)

# dE/dx_i = sum_{j!=i} (-m T_ij / d_ij^2) * 2(x_i - x_j), T_ij=(lam/d_ij^2)^m;
# chain through the sine: dE/dx_tilde_i = (dE/dx_i) * cos(x_tilde_i).
def objective_grad(u, n, sharpness, scale):
    ut = u.reshape(2, n)
    x, y = np.sin(ut[0]), np.sin(ut[1])
    dx = x[:, None] - x[None, :]; dy = y[:, None] - y[None, :]
    d2 = dx*dx + dy*dy
    np.fill_diagonal(d2, np.inf)
    T = (scale / d2)**sharpness
    coef = -sharpness * T / d2                     # scale is fixed during this stage
    gx = np.sum(coef * 2*dx, axis=1)
    gy = np.sum(coef * 2*dy, axis=1)
    return np.concatenate([gx * np.cos(ut[0]), gy * np.cos(ut[1])])

def detect_contacts(points, distance_tol=1e-7, wall_tol=1e-7):
    x, y = points[:, 0], points[:, 1]
    d2 = pairwise_sq_dists(x, y)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2.min())
    iu = np.triu_indices(len(points), 1)
    distances = np.sqrt(d2[iu])
    pairs = [(i, j) for i, j, dij in zip(iu[0], iu[1], distances) if abs(dij - d) <= distance_tol]
    walls = []
    for i, (px, py) in enumerate(points):
        if abs(px + 1.0) <= wall_tol:
            walls.append((i, 0, -1.0))
        if abs(px - 1.0) <= wall_tol:
            walls.append((i, 0, 1.0))
        if abs(py + 1.0) <= wall_tol:
            walls.append((i, 1, -1.0))
        if abs(py - 1.0) <= wall_tol:
            walls.append((i, 1, 1.0))
    return d, pairs, walls

def polish_contacts(points, distance_tol=1e-7, wall_tol=1e-7):
    n = len(points)
    d0, pairs, walls = detect_contacts(points, distance_tol, wall_tol)
    z0 = np.concatenate([points[:, 0], points[:, 1], [d0]])

    def residual(z):
        x, y, d = z[:n], z[n:2*n], z[-1]
        out = [((x[i]-x[j])**2 + (y[i]-y[j])**2 - d*d) for i, j in pairs]
        for i, axis, value in walls:
            out.append((x if axis == 0 else y)[i] - value)
        return np.asarray(out)

    if len(pairs) + len(walls) == 0:
        return points, d0
    lo = np.r_[-np.ones(2*n), 0.0]
    hi = np.r_[ np.ones(2*n), np.inf]
    res = least_squares(residual, z0, bounds=(lo, hi),
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
    z = res.x
    return np.column_stack([z[:n], z[n:2*n]]), z[-1]

def optimize_one_start(u, n, schedule=None):
    if schedule is None:
        schedule = (10, 20, 40, 80, 160, 320, 640, 1280)
    for sharpness in schedule:                         # homotopy: soft energy -> sharp max-min
        d = min_pair_distance(u, n)
        scale = d*d if d > 0 else 1.0                 # lambda = d_min^2, fixed in this stage
        res = minimize(objective, u, args=(n, sharpness, scale), jac=objective_grad,
                       method='CG', options={'maxiter': 500, 'gtol': 1e-10})
        u = res.x
    return u

def solve(n, restarts=50, rng=None, schedule=None):
    rng = rng or np.random.default_rng(0)
    best_u, best_d_side2 = None, -1.0
    for _ in range(restarts):                         # cover the basins
        u = rng.uniform(-np.pi/2, np.pi/2, 2*n)
        u = optimize_one_start(u, n, schedule)
        d_side2 = min_pair_distance(u, n)
        if d_side2 > best_d_side2:
            best_d_side2, best_u = d_side2, u.copy()
    x, y = to_box(best_u, n)
    polished, polished_d_side2 = polish_contacts(np.column_stack([x, y]))
    return polished_d_side2 / 2.0, (polished + 1.0) / 2.0, best_u
```
