The problem is to place n equal, non-overlapping circles inside the unit square so that their common radius is as large as possible. After absorbing the wall offset, this is the same as placing n points in a unit square to maximize the minimum pairwise distance d_n; the circle radius is then r_n = d_n / (2(d_n + 1)). The difficulty is that d_n is a minimum over all pairs, so the objective is non-smooth and kinked. Smooth quasi-Newton methods such as BFGS stall at these kinks, which is why direct radius maximization fails even for modest n like 14, 15, and 17. Brute-force grid search or contact-graph enumeration is also the wrong instinct: the coordinates are continuous, a fine grid is exponential in 2n, and the best packings often move by tiny symmetry-breaking offsets that no grid captures. A method needs to smooth the non-smooth objective, handle the box constraint cheaply, and escape the many local optima.

The method I propose is energy-function minimization with exponent annealing. The hard minimum over pairs is replaced by a smooth surrogate using the negative-power mean: min_{i<j} ||s_i - s_j|| is the limit as p goes to negative infinity of (sum_{i<j} ||s_i - s_j||^p)^{1/p}. For fixed p < 0, maximizing this soft minimum is the same as minimizing the sum of p-th powers, which for p = -2m becomes an inverse-power repulsion energy E = sum_{i<j} (lambda / d_ij^2)^m, where d_ij^2 is the squared Euclidean distance. As the sharpness parameter m grows, the sum is dominated by the single closest pair, so minimizing E increasingly pushes up the minimum distance and recovers the max-min objective in the limit.

To make this practical, the box constraint is removed by a sine parameterization: x_i = sin(x_tilde_i) and y_i = sin(y_tilde_i), so the search over the combined vector u = (x_tilde, y_tilde) in R^{2n} is fully unconstrained and the coordinates always stay inside the square. The gradient chains through cos(x_tilde_i). The raw power sum is numerically unstable for large m, so the implementation minimizes log E computed with logsumexp; since log is monotone the minimizers are unchanged, but overflow is avoided. Within each stage the scale factor lambda is set to the square of the current minimum distance and held fixed, so the dominant term starts near one and the arithmetic stays well-conditioned.

The key to making the method both tractable and faithful is exponent annealing, which is a homotopy. Small m gives a smooth, forgiving energy that is easy to minimize but rewards spreading all distances rather than only the worst one; large m gives a faithful proxy for the minimum distance but a viciously multimodal landscape. The algorithm starts with a small m, minimizes, then doubles m and re-minimizes warm-started from the previous solution. Repeating this tracks the minimizer as the objective morphs from the easy soft version into the true max-min. A compact double-precision schedule such as (10, 20, 40, 80, 160, 320, 640, 1280) is usually enough, while a high-precision search can continue toward m around 10^6 or even 10^50 for stubborn cases.

Because the landscape still has many local optima, the algorithm uses many random restarts per n, keeping the best configuration. A small repair step is also included: if a point is pinned against a wall but could slide inward without reducing the minimum distance, it is nudged inward and re-minimized. Finally, the structure is polished by detecting contacts: pairs at the minimum distance and points on walls form a nonlinear system that is solved numerically to sharpen the reported d_n. The compact code below accepts the polished result only if the contact residual is small and the actual minimum distance agrees with the solved contact distance.

```python
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.special import logsumexp

def to_box(u, n):
    ut = u.reshape(2, n)
    return np.sin(ut[0]), np.sin(ut[1])

def points_to_u(points):
    points = np.clip(points, -1.0, 1.0)
    return np.concatenate([np.arcsin(points[:, 0]), np.arcsin(points[:, 1])])

def pairwise_sq_dists(x, y):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return dx * dx + dy * dy

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

def objective(u, n, sharpness, scale):
    x, y = to_box(u, n)
    d2 = pairwise_sq_dists(x, y)
    iu = np.triu_indices(n, 1)
    tiny = np.finfo(float).tiny
    log_scale = np.log(max(scale, tiny))
    log_terms = sharpness * (log_scale - np.log(np.maximum(d2[iu], tiny)))
    return logsumexp(log_terms)

def objective_grad(u, n, sharpness, scale):
    ut = u.reshape(2, n)
    x, y = np.sin(ut[0]), np.sin(ut[1])
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    np.fill_diagonal(d2, np.inf)
    tiny = np.finfo(float).tiny
    iu = np.triu_indices(n, 1)
    safe_pair_d2 = np.maximum(d2[iu], tiny)
    log_scale = np.log(max(scale, tiny))
    log_terms = sharpness * (log_scale - np.log(safe_pair_d2))
    log_z = logsumexp(log_terms)
    weights = np.exp(log_terms - log_z)
    coef = -sharpness * weights / safe_pair_d2
    fx = coef * 2 * dx[iu]
    fy = coef * 2 * dy[iu]
    gx = np.zeros(n)
    gy = np.zeros(n)
    np.add.at(gx, iu[0], fx)
    np.add.at(gx, iu[1], -fx)
    np.add.at(gy, iu[0], fy)
    np.add.at(gy, iu[1], -fy)
    return np.concatenate([gx * np.cos(ut[0]), gy * np.cos(ut[1])])

def maybe_release_wall_points(u, n, wall_tol=1e-6, inward_step=1e-3, keep_tol=1e-10):
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
        out = [((x[i] - x[j])**2 + (y[i] - y[j])**2 - d*d) for i, j in pairs]
        for i, axis, value in walls:
            out.append((x if axis == 0 else y)[i] - value)
        return np.asarray(out)

    if len(pairs) + len(walls) == 0:
        return points, d0
    lo = np.r_[-np.ones(2*n), 0.0]
    hi = np.r_[np.ones(2*n), np.inf]
    res = least_squares(residual, z0, bounds=(lo, hi),
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
    z = res.x
    solved_d = z[-1]
    polished = np.column_stack([z[:n], z[n:2*n]])
    actual_d = min_pair_distance_from_points(polished)
    residual_norm = np.linalg.norm(residual(z), ord=2)
    residual_tol = max(1e-8, 1e-6 * d0 * d0)
    distance_tol = max(1e-7, 1e-6 * max(1.0, d0))
    if (not res.success) or (not np.isfinite(actual_d)) or (not np.isfinite(solved_d)) or residual_norm > residual_tol:
        return points, d0
    if actual_d < d0 * (1.0 - 1e-8):
        return points, d0
    if abs(actual_d - solved_d) > distance_tol:
        return points, d0
    return polished, actual_d

def optimize_one_start(u, n, schedule=None):
    if schedule is None:
        schedule = (10, 20, 40, 80, 160, 320, 640, 1280)
    for sharpness in schedule:
        d = min_pair_distance(u, n)
        scale = d * d if d > 0 else 1.0
        res = minimize(objective, u, args=(n, sharpness, scale), jac=objective_grad,
                       method='CG', options={'maxiter': 500, 'gtol': 1e-10})
        u = res.x
        u, released = maybe_release_wall_points(u, n)
        if released:
            d = min_pair_distance(u, n)
            scale = d * d if d > 0 else 1.0
            res = minimize(objective, u, args=(n, sharpness, scale), jac=objective_grad,
                           method='CG', options={'maxiter': 500, 'gtol': 1e-10})
            u = res.x
    return u

def solve(n, restarts=50, rng=None, schedule=None):
    if n == 1:
        return np.inf, np.array([[0.5, 0.5]]), np.zeros(2)
    rng = rng or np.random.default_rng(0)
    best_u, best_d_side2 = None, -1.0
    for _ in range(restarts):
        u = rng.uniform(-np.pi / 2, np.pi / 2, 2 * n)
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
