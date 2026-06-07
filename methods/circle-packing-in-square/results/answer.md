# Energy-function minimization with exponent annealing for packing equal circles in a square

## Problem

Place `n` equal non-overlapping circles in the unit square to maximize the common radius `r_n`.
Absorbing the wall offset and rescaling turns this into the point-spreading problem: place `n`
points in a unit square to maximize the minimum pairwise distance `d_n = min_{i<j} ‖s_i − s_j‖`.
If the centre square has side `1 − 2r`, rescaling it to side `1` turns a nearest centre spacing
`2r` into `d = 2r/(1 − 2r)`, so `d(1 − 2r) = 2r` and

    r_n = d_n / (2 (d_n + 1)).

The optimizer below returns this unit-square distance `d_n` as its first value. The circle radius
is a separate conversion through the formula above.

The objective `d_n` is a non-smooth `min`-over-pairs, multimodal in the `2n` coordinates. That is
what defeats naive smooth optimizers (direct radius maximization with simplex/BFGS fails for
`n = 14, 15, 17` already), and it also makes brute-force discretization the wrong instinct: a grid
search is exponential in dimension, misses off-grid broken-symmetry packings, and still has to
guess the active contact graph.

## Key idea

Relax the hard minimum into a smooth surrogate via the negative-power mean
`min_{i<j} ‖s_i−s_j‖ = lim_{p→−∞} (Σ_{i<j} ‖s_i−s_j‖^p)^{1/p}`. For fixed `p < 0`, the outer
`1/p` power is decreasing, so maximizing this soft minimum is the same as minimizing
`Σ ‖s_i−s_j‖^p`. The implementation works with squared distances
`d_ij² = (x_i − x_j)² + (y_i − y_j)²`, so set `p = −2m`; the positive sharpness `m` is half the
negative Euclidean-distance exponent. Multiplying by fixed `λ^m` within a stage gives the
repulsion energy:

    E = Σ_{1≤i<j≤n} ( λ / d_ij² )^m,        d_ij² = (x_i − x_j)² + (y_i − y_j)².

As `m` grows the sum becomes dominated by the single closest pair, so minimizing `E` increasingly
pushes up the minimum distance and recovers the max-min objective in the limit; in
Euclidean-distance terms this is a `1/ρ^{2m}` repulsion. Three supports make this practical:

1. **Unconstrained box via the sine map.** Set `x_i = sin(x̃_i)`, `y_i = sin(ỹ_i)` with
   `x̃_i, ỹ_i ∈ R`. Then `|x_i|, |y_i| ≤ 1` automatically (a side-2 box; the unit square up to a
   harmless rescale), so the search over `u = (x̃, ỹ) ∈ R^{2n}` is fully unconstrained. Gradients
   chain through `∂x_i/∂x̃_i = cos(x̃_i)`.

2. **Scale factor and log energy.** With huge `m`, `(λ/d²)^m` overflows or underflows unless the
   base sits near 1. After each stage set `λ = d²_min`, the square of the current shortest
   distance, so the dominant term starts at `1`. `λ` is held fixed inside that stage's objective
   and gradient; it conditions the raw arithmetic only. In the runnable double-precision code,
   minimize `log E = log Σ exp(m(log λ − log d_ij²))` using `logsumexp`. Since `log` is monotone,
   the minimizers are the same as for `E`, while trial steps that would overflow the raw power sum
   stay finite.

3. **Exponent annealing (homotopy).** Small `m` ⇒ smooth, easy, but a poor proxy (rewards spreading
   all distances). Large `m` ⇒ faithful proxy, but viciously multimodal. Start small (`m` in
   10–100), minimize, then repeatedly double `m`, recompute `λ`, and re-minimize warm-started from
   the previous solution — tracking the minimizer as the objective deforms from soft to sharp. The
   high-precision search can continue toward `m ≈ 10⁶`, and stubborn cases can be pushed as far as
   about `10⁵⁰`; the compact code defaults to `(10, 20, ..., 1280)` because that is already sharp
   enough for small sanity checks in ordinary double precision, and a longer schedule can be passed
   explicitly. Use many random restarts (`≥ 50` per `n`) to cover the basins. If a point is pinned
   against a wall without being held there by the current gap, try a tiny inward move, accept it
   only when the minimum distance does not drop, and re-minimize at that exponent.

## Final algorithm

For each `n`: (a) repeat `≥ 50` times from random starts; (b) within each start, anneal `m` upward,
at each stage setting `λ = d²_min` and minimizing `E` (implemented as `log E` in the runnable code)
with an analytic gradient; the full search can use steepest descent plus modified Newton, while the
compact code below uses SciPy's conjugate-gradient minimizer; (c) keep the best configuration; (d)
**refine**: sort all pairwise distances, find the jump separating contacts (≈ `d_n`) from gaps,
treat near-wall points as on-wall, form the resulting nonlinear contact system, and solve it
numerically. The compact code accepts this polish only if the contact residual is small, the solved
contact distance is finite and agrees with the actual minimum pairwise distance, and that actual
minimum has not decreased; otherwise it keeps the optimizer output. In the high-precision version,
the same contact system is solved in arbitrary precision and, in special structured cases, can be
reduced to a univariate polynomial for the diameter.

Gradient for the raw energy with `λ` held fixed (`T_ij = (λ/d_ij²)^m`):

    ∂E/∂x_i = Σ_{j≠i} (−m T_ij / d_ij²) · 2(x_i − x_j),
    ∂E/∂x̃_i = (∂E/∂x_i) · cos(x̃_i),

and likewise for `y`. The code minimizes `log E`, so it uses the same force with normalized
weights `w_ij = T_ij / Σ_{a<b} T_ab`:

    ∂logE/∂x_i = Σ_{j≠i} (−m w_ij / d_ij²) · 2(x_i − x_j).

## Code

```python
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.special import logsumexp

# x_i = sin(x_tilde_i), y_i = sin(y_tilde_i): the box constraint |x|,|y| <= 1 is automatic,
# so the search over u = [x_tilde; y_tilde] in R^{2n} is unconstrained.
def to_box(u, n):
    ut = u.reshape(2, n)
    return np.sin(ut[0]), np.sin(ut[1])              # coords in [-1, 1] (a side-2 box)

def points_to_u(points):
    points = np.clip(points, -1.0, 1.0)
    return np.concatenate([np.arcsin(points[:, 0]), np.arcsin(points[:, 1])])

def pairwise_sq_dists(x, y):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return dx*dx + dy*dy

def min_pair_distance_from_points(points):
    if len(points) < 2:
        return np.inf
    d2 = pairwise_sq_dists(points[:, 0], points[:, 1])
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2.min())

def min_pair_distance(u, n):
    x, y = to_box(u, n)
    return min_pair_distance_from_points(np.column_stack([x, y]))

def radius_from_unit_distance(d_unit):
    if np.isinf(d_unit):
        return 0.5
    return d_unit / (2.0 * (1.0 + d_unit))

# Minimize log(E), E = sum_{i<j} (lambda / d_ij^2)^m. log is monotone, so the
# minimizers are unchanged, and logsumexp avoids overflow at large m.
def objective(u, n, sharpness, scale):
    x, y = to_box(u, n)
    d2 = pairwise_sq_dists(x, y)
    iu = np.triu_indices(n, 1)
    tiny = np.finfo(float).tiny
    log_scale = np.log(max(scale, tiny))
    log_terms = sharpness * (log_scale - np.log(np.maximum(d2[iu], tiny)))
    return logsumexp(log_terms)

# d log(E)/dx_i = sum_{j!=i} (-m w_ij / d_ij^2) * 2(x_i - x_j),
# w_ij = T_ij / sum_{a<b} T_ab. Chain through the sine map.
def objective_grad(u, n, sharpness, scale):
    ut = u.reshape(2, n)
    x, y = np.sin(ut[0]), np.sin(ut[1])
    dx = x[:, None] - x[None, :]; dy = y[:, None] - y[None, :]
    d2 = dx*dx + dy*dy
    np.fill_diagonal(d2, np.inf)
    tiny = np.finfo(float).tiny
    iu = np.triu_indices(n, 1)
    safe_pair_d2 = np.maximum(d2[iu], tiny)
    log_scale = np.log(max(scale, tiny))
    log_terms = sharpness * (log_scale - np.log(safe_pair_d2))
    log_z = logsumexp(log_terms)
    weights = np.exp(log_terms - log_z)             # one normalized weight per unordered pair
    coef = -sharpness * weights / safe_pair_d2
    fx = coef * 2*dx[iu]
    fy = coef * 2*dy[iu]
    gx = np.zeros(n)
    gy = np.zeros(n)
    np.add.at(gx, iu[0], fx)
    np.add.at(gx, iu[1], -fx)
    np.add.at(gy, iu[0], fy)
    np.add.at(gy, iu[1], -fy)
    return np.concatenate([gx * np.cos(ut[0]), gy * np.cos(ut[1])])

def maybe_release_wall_points(u, n, wall_tol=1e-6, inward_step=1e-3, keep_tol=1e-10):
    """Move a non-rigid wall point slightly inward if that does not reduce the current gap."""
    x, y = to_box(u, n)
    points = np.column_stack([x, y])
    current = min_pair_distance_from_points(points)
    changed = False
    for i in range(n):
        for axis in (0, 1):
            value = points[i, axis]
            if 1.0 - abs(value) > wall_tol:
                continue
            trial = points.copy()
            trial[i, axis] = value - np.sign(value) * inward_step
            trial[i, axis] = np.clip(trial[i, axis], -1.0, 1.0)
            trial_d = min_pair_distance_from_points(trial)
            if trial_d >= current * (1.0 - keep_tol):
                points = trial
                current = trial_d
                changed = True
    return (points_to_u(points), True) if changed else (u, False)

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
    solved_d = z[-1]
    polished = np.column_stack([z[:n], z[n:2*n]])
    actual_d = min_pair_distance_from_points(polished)
    residual_norm = np.linalg.norm(residual(z), ord=2)
    residual_tol = max(1e-8, 1e-6*d0*d0)
    distance_tol = max(1e-7, 1e-6*max(1.0, d0))
    if (
        (not res.success)
        or (not np.isfinite(actual_d))
        or (not np.isfinite(solved_d))
        or residual_norm > residual_tol
    ):
        return points, d0
    if actual_d < d0 * (1.0 - 1e-8):
        return points, d0
    if abs(actual_d - solved_d) > distance_tol:
        return points, d0
    return polished, actual_d

def optimize_one_start(u, n, schedule=None):
    if schedule is None:
        schedule = (10, 20, 40, 80, 160, 320, 640, 1280)
    for sharpness in schedule:                         # homotopy: soft energy -> sharp max-min
        d = min_pair_distance(u, n)
        scale = d*d if d > 0 else 1.0                 # lambda = d_min^2, fixed in this stage
        res = minimize(objective, u, args=(n, sharpness, scale), jac=objective_grad,
                       method='CG', options={'maxiter': 500, 'gtol': 1e-10})
        u = res.x
        u, released = maybe_release_wall_points(u, n)
        if released:
            d = min_pair_distance(u, n)
            scale = d*d if d > 0 else 1.0
            res = minimize(objective, u, args=(n, sharpness, scale), jac=objective_grad,
                           method='CG', options={'maxiter': 500, 'gtol': 1e-10})
            u = res.x
    return u

def solve(n, restarts=50, rng=None, schedule=None):
    """Return (d_unit, points_unit, raw_variables).

    d_unit is the minimum pairwise distance after mapping the side-2 optimizer
    box to the unit square. Convert it to circle radius with
    radius_from_unit_distance(d_unit).
    """
    if n == 1:
        return np.inf, np.array([[0.5, 0.5]]), np.zeros(2)
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
    d_unit = polished_d_side2 / 2.0
    points_unit = (polished + 1.0) / 2.0
    return d_unit, points_unit, points_to_u(polished)
```
