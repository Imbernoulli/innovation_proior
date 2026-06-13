# k-means (Lloyd's iteration + k-means++ seeding), distilled

k-means partitions `n` points in `R^d` into `k` clusters by minimizing the within-cluster sum of
squares (inertia). It does this with a parameter-free local search — alternate between assigning each
point to its nearest center and moving each center to the mean of its cluster — that provably decreases
the objective every step and converges to a local minimum. Because that local minimum is hostage to the
initialization, k-means++ seeds the centers by sampling each new center with probability proportional
to its squared distance to the nearest already-chosen center (`D^2` weighting), which spreads the
centers, resists outliers, and gives an `O(log k)` approximation guarantee that the bare local search
lacks.

## Problem it solves

Given points `X = {x_1, ..., x_n} ⊂ R^d` and an integer `k`, choose `k` centers `C = {c_1, ..., c_k}`
minimizing

```
phi(C) = sum_{x in X} min_{c in C} || x - c ||^2     (inertia / within-cluster sum of squares).
```

The centers implicitly define the clustering (each point joins its nearest center). Minimizing `phi`
exactly is NP-hard even for `k = 2`, and `phi` is non-convex in the center locations, so the practical
goal is a fast local search plus a seeding that guarantees a good basin.

## Key idea

**Two forced steps (alternating / coordinate minimization).** `phi` couples assignments and center
locations; minimize one with the other fixed.

- *Assignment step:* with centers fixed, each point's term `min_c || x - c ||^2` is minimized by
  assigning it to its nearest center (Voronoi cells). Exactly optimal, term by term.
- *Update step:* with the assignment fixed, the best center for a cluster `S` minimizes
  `sum_{x in S} || x - z ||^2`. By the **parallel-axis identity**
  `sum_{x in S} || x - z ||^2 = sum_{x in S} || x - c(S) ||^2 + |S| || c(S) - z ||^2`, the unique
  minimizer is the mean `c(S)`. (This is also why the objective is squared, not absolute: squared error's
  optimal representative is the cheap closed-form mean; absolute error's is the median.)

Both steps weakly decrease `phi`; `phi >= 0` and there are finitely many partitions, so the loop
converges to a fixed point in finitely many sweeps — but only a local minimum.

**k-means++ seeding (`D^2` weighting).** Uniform random seeds give an unbounded `phi / phi_OPT`
(centers clump in one group, miss others). Deterministic farthest-point seeding spreads centers but
plants them on outliers. The fix: randomize farthest-point — pick the first center uniformly, then
sample each next center with probability `D(x)^2 / sum_y D(y)^2`, where `D(x)` is the distance from `x`
to its nearest already-chosen center. The square matches the squared-error objective (`D(x)^2` is `x`'s
contribution to `phi`); randomization makes a lone outlier just one low-mass point.

## Guarantee

Let `phi_OPT` be the optimal potential, `phi(A)`/`phi_OPT(A)` a set `A`'s contributions.

- **Uniform seed of an optimal cluster `A`:** `E[phi(A)] = 2 phi_OPT(A)` (parallel-axis identity).
- **`D^2` seed landing in `A`:** `E[phi(A)] <= 8 phi_OPT(A)`, via `D(a_0) <= D(a) + ||a - a_0||`
  (triangle) and `(p+q)^2 <= 2p^2 + 2q^2` (power-mean), then averaging over `A` and reusing the
  uniform-seed computation.
- **Full seeding** (first center uniform, rest `D^2`): by induction over (centers left to place `t`,
  uncovered optimal clusters `u`), `E[phi'] <= (phi(X_c) + 8 phi_OPT(X_u))(1 + H_t) + ((u-t)/u)phi(X_u)`
  with `H_t = 1 + 1/2 + ... + 1/t`; specializing `t = u = k - 1` and `H_{k-1} <= 1 + ln k`:

```
E[phi] <= 8 (ln k + 2) phi_OPT .
```

The subsequent Lloyd local search only decreases `phi`, so the combined method inherits the `O(log k)`
bound for *any* data set, no separation assumption. The `log k` is tight: on a simplex-of-simplices
instance `D^2` seeding is no better than `2 ln k`-competitive.

**Generalization.** For `phi^[l] = sum_x min_c || x - c ||^l` (`l = 1` is k-median), `D^l` weighting
gives `E[phi^[l]] <= 2^{2l}(ln k + 2) phi_OPT^[l]`; the seeding exponent is locked to the objective's.

## Practical choices and why

- **Greedy seeding:** at each step draw `n_local_trials = 2 + int(log k)` candidate points by `D^2`,
  keep the one that most reduces `phi` — cheap variance reduction on the random seed.
- **Restarts:** `phi` is non-convex, so run the whole procedure `n_init = 10` times from
  independent seedings and keep the lowest-`phi` run.
- **Empty clusters:** a center that wins no points has an undefined mean; relocate it to the point with
  the largest `D^2` (farthest from its center) — the same far-from-everything criterion as seeding.
- **`max_iter = 300`** caps the local loop; strict label stability stops immediately, center-shift
  tolerance can also stop it, and a final assignment refresh keeps labels aligned with centers.

## Final algorithm

```
# Seeding (k-means++):
c_1            <- uniform random point from X
for c in 2..k:
    D(x)^2     <- min over chosen centers of ||x - center||^2   for all x
    draw 2 + int(log k) candidates ~ D(x)^2;  keep the one minimizing sum_x min(D(x)^2, ||x-cand||^2)

# Refinement (Lloyd), repeated for n_init seedings, keep lowest inertia:
repeat until labels unchanged, center shift <= tol, or max_iter:
    assign each x to its nearest center                          # Voronoi step
    move each center to the mean of its assigned points          # centroid step
    (reseed any empty center at the point with the largest D^2)
if stopped without strict label convergence, refresh assignments
inertia = sum_x min_c ||x - c||^2
```

## Working code

```python
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


def _sq_dist_to_nearest(X, centers):
    """For each point: squared Euclidean distance to its nearest center, and which one."""
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)   # (n, k)
    return d2.min(axis=1), d2.argmin(axis=1)


def _kmeanspp_init(X, k, rng):
    """Seed k centers by D^2 weighting, greedily keeping the best of a few candidates."""
    n = X.shape[0]
    n_local_trials = 2 + int(np.log(k))
    centers = np.empty((k, X.shape[1]), dtype=X.dtype)
    indices = np.full(k, -1, dtype=int)

    center_id = rng.choice(n)                              # first center: uniform at random
    centers[0] = X[center_id]
    indices[0] = center_id
    closest_d2 = ((X - centers[0]) ** 2).sum(axis=1)       # D(x)^2 to the chosen center
    current_pot = closest_d2.sum()                         # phi = sum_x D(x)^2

    for c in range(1, k):
        rand_vals = rng.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(np.cumsum(closest_d2), rand_vals)
        np.clip(candidate_ids, None, n - 1, out=candidate_ids)
        distance_to_candidates = ((X[candidate_ids, None, :] - X[None, :, :]) ** 2).sum(axis=2)
        np.minimum(closest_d2, distance_to_candidates, out=distance_to_candidates)
        candidate_pot = distance_to_candidates.sum(axis=1)
        best = np.argmin(candidate_pot)                    # greedily keep the phi-reducing candidate
        current_pot = candidate_pot[best]
        closest_d2 = distance_to_candidates[best]
        best_candidate = candidate_ids[best]
        centers[c] = X[best_candidate]
        indices[c] = best_candidate
    return centers, indices


def _centers_from_labels(X, labels, centers, d2):
    """Move nonempty clusters to their means; relocate empty clusters to far points."""
    k = centers.shape[0]
    new_centers = centers.copy()
    counts = np.bincount(labels, minlength=k)
    for j in range(k):
        if counts[j] > 0:
            new_centers[j] = X[labels == j].mean(axis=0)              # centroid step
    empty = np.where(counts == 0)[0]
    if len(empty) > 0:
        farthest = np.argsort(d2)[::-1]
        for j, point_id in zip(empty, farthest):
            new_centers[j] = X[point_id]                              # empty cluster relocation
    return new_centers


def _lloyd(X, centers, max_iter=300, tol=1e-4):
    """Batch Lloyd iteration with strict-label and center-shift stopping."""
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    strict_convergence = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        d2, labels = _sq_dist_to_nearest(X, centers)                  # assignment step
        new_centers = _centers_from_labels(X, labels, centers, d2)
        center_shift_tot = ((new_centers - centers) ** 2).sum()
        centers = new_centers

        if np.array_equal(labels, labels_old):
            strict_convergence = True
            break
        if center_shift_tot <= tol:
            break
        labels_old[:] = labels

    if not strict_convergence:
        d2, labels = _sq_dist_to_nearest(X, centers)                  # final E-step after tol/max_iter
    else:
        d2 = ((X - centers[labels]) ** 2).sum(axis=1)
    inertia = d2.sum()
    return centers, labels, inertia, n_iter


class CustomClustering(BaseEstimator, ClusterMixin):
    """k-means: k-means++ seeding + Lloyd refinement, best of n_init restarts."""

    def __init__(self, n_clusters=None, random_state=42, n_init=10, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        k = self.n_clusters if self.n_clusters is not None else 8
        if k > X.shape[0]:
            raise ValueError("n_clusters must be no larger than n_samples")
        rng = np.random.RandomState(self.random_state)
        best_inertia, best_centers, best_labels, best_n_iter = np.inf, None, None, 0
        for _ in range(self.n_init):                                  # non-convex: hedge with restarts
            centers, _ = _kmeanspp_init(X, k, rng)
            centers, labels, inertia, n_iter = _lloyd(X, centers, self.max_iter, self.tol)
            if inertia < best_inertia:                               # keep the lowest-phi run
                best_inertia = inertia
                best_centers, best_labels, best_n_iter = centers, labels, n_iter
        self.cluster_centers_, self.labels_ = best_centers, best_labels
        self.inertia_, self.n_iter_ = best_inertia, best_n_iter
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            self.fit(X)
        return _sq_dist_to_nearest(np.asarray(X, dtype=float), self.cluster_centers_)[1]
```
