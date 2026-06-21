# Context: partitioning unlabeled points by squared-distance compactness (circa mid-2000s)

## Research question

I am given `n` points `X = {x_1, ..., x_n}` in `R^d` and an integer `k`, and no labels at all. I want
to split the points into `k` groups so that points in the same group are close together and points in
different groups are far apart — to recover whatever latent structure produced the data. To make
"close together" precise I need a single scalar that scores a candidate set of `k` representative
points (centers) `C = {c_1, ..., c_k}`: the natural one is the total squared distance from each point
to its nearest center,

```
phi(C) = sum_{x in X} min_{c in C} || x - c ||^2 .
```

A small `phi` means every point sits near some center, i.e. the centers compactly summarize the data;
the center set implicitly defines the clustering (each point joins its nearest center). The goal is to
choose `C` to make `phi` small. Minimizing `phi` exactly is NP-hard — even for `k = 2` clusters in the
plane — so for any realistic `n, k, d` an exact solution is out of reach and I am forced to a heuristic.
`phi` viewed as a function of the center locations is not convex: it is a minimum (over centers) of
convex pieces, so a local-search heuristic settles into a fixed point that depends on its starting
configuration. The practical question is how to initialize and run such a search.

## Background

Clustering — grouping unlabeled data so the groups reflect latent structure — is one of the oldest
problems in statistics and pattern recognition, and the squared-error formulation above is the most
classical. The load-bearing concepts:

**Within-class variance as the objective.** The squared-error / within-cluster-sum-of-squares
criterion, `W = sum_i integral_{S_i} || z - x_i ||^2 dp(z)` for a partition `{S_i}` with
representatives `{x_i}`, is the dominant compactness score (MacQueen, 1967, who named the resulting
method). It is the empirical analogue of the variance of a distribution, decomposed across groups. The
reason squared error is the workhorse rather than, say, absolute error is a closed-form fact: for a
fixed set of points, the single representative `z` minimizing `sum || x - z ||^2` is their arithmetic
mean (the "center of mass"), whereas for absolute error it is the (coordinatewise) median, which has no
such clean linear form. The mean's optimality under squared error is the **parallel-axis / moment-of-
inertia identity**: a set's total squared distance to an arbitrary point `z` equals its total squared
distance to its own center of mass plus `|S|` times the squared offset of `z` from that center — so the
moment of inertia of a body of mass is minimized about its center of mass (a classical result; e.g.
Lloyd, 1982, citing it as standard). This identity is the analytic engine behind everything that
follows.

**Voronoi assignment.** Given a fixed set of centers, the partition of space that minimizes squared
error assigns each point to its nearest center; the resulting regions are Voronoi cells whose
boundaries bisect the segments between adjacent centers (Lloyd, 1982, conditions on the optimal
quantizer). Assignment-to-nearest is forced, not chosen: each point's contribution to `phi` is a `min`
over centers, achieved pointwise by the closest one.

**The quantization heritage.** The exact same objective and the exact same two conditions — each
representative is the centroid of its cell, each cell is the nearest-representative region — arise in
signal quantization, where one must place a finite set of "quanta" to minimize the mean squared
quantization noise of a source (Lloyd, 1982; presented 1957). There, the noise power
`N = sum_alpha integral_{Q_alpha} (q_alpha - x)^2 dF(x)` is structurally identical to `phi`, and the
two necessary conditions for an optimum are (i) `q_alpha = ` center of mass of cell `Q_alpha` and (ii)
the cells are the nearest-quantum regions with midpoint boundaries. So the compactness objective and
its two stationarity conditions were understood decades before they were a clustering tool.

**Properties of the squared-error objective.** The objective is non-convex in the center locations, so
where the search starts determines where it ends — the same data with two different initializations
yields two different clusterings. Starting centers uniformly at random from the data can place several
initial centers inside one true group while leaving other true groups with no nearby center. The
squared-error / Euclidean-mean criterion implicitly assumes clusters are roughly convex and isotropic
(equal-spread balls), and in high dimension Euclidean distances concentrate (all pairwise distances
become similar), eroding the contrast the assignment step relies on — a known reason to reduce
dimension (e.g. by PCA) before clustering.

## Baselines

The prior methods a new clustering procedure would be measured against and would react to.

**The alternating assignment/centroid loop — Forgy (1965), Jennrich; Lloyd's quantizer iteration
(1982).** The canonical local search for `phi`. Start from some `k` centers; repeat two steps until the
assignment stops changing: (1) assign every point to its nearest current center (the Voronoi step);
(2) move each center to the mean of the points now assigned to it (the centroid step). Each step weakly
decreases `phi` — the assignment step by the pointwise `min`, the centroid step by the parallel-axis
identity — and because `phi >= 0` and there are only finitely many partitions, the loop terminates at a
fixed point in finitely many sweeps. It is simple, near-linear per sweep, and usually converges in few
sweeps. In quantizer language this is Lloyd's "Method I": impose the centroid condition and the
midpoint-partition condition alternately, producing a monotonically decreasing noise sequence
`N(rho^(1)) >= N(rho^(2)) >= ...`.

**Uniform random seeding.** The default way to start the loop: pick the `k` initial centers uniformly
at random from the data points.

**Deterministic farthest-point / maximin seeding.** A natural way to spread centers: pick the first
center arbitrarily, then repeatedly add the data point that is farthest from the centers chosen so far.

**The online / sequential mean update (MacQueen, 1967).** The original `k`-means as MacQueen defined
it processes points one at a time: keep running means and counts for `k` groups; for each incoming
point, assign it to the nearest current mean and update that mean incrementally,
`x_i <- (x_i * w_i + z) / (w_i + 1)`, `w_i <- w_i + 1`. The name "`k`-means" comes from this: at every
stage the `k` representatives are literally the means of the groups they currently represent, a
generalization of the ordinary sample mean. MacQueen proved asymptotic convergence of the within-class
variance for this sequential process.

## Evaluation settings

The natural yardsticks already in use for clustering, all pre-existing:

- **Synthetic isotropic Gaussian blobs** — several well-separated spherical Gaussian clusters in
  moderate dimension; the friendly case the squared-error objective is built for, used to check that a
  method recovers the planted structure.
- **Non-convex shapes** — two interleaving half-moons / concentric rings in the plane; clusters that
  are not linearly or convexly separable, used to expose the convex-isotropic assumption.
- **High-dimensional real data** — e.g. handwritten-digit feature vectors (`load_digits`, 64 features,
  10 classes); real, high-dimensional, many-class data where distance concentration bites.
- Metrics computed against held-out ground-truth labels and intrinsically: **Adjusted Rand Index**
  (chance-corrected agreement of predicted vs. true partition), **Normalized Mutual Information**
  (information shared between the two labelings, normalized to `[0,1]`), and **silhouette** (per-point
  mean intra-cluster vs. nearest-other-cluster distance, an intrinsic compactness/separation score).
- Protocol: the number of clusters `k` is supplied; because the objective is non-convex, the search is
  commonly run several times from independent initializations and the lowest-`phi` run is kept, capped
  by a maximum iteration count per run.

## Code framework

The procedure plugs into a standard estimator harness: a class with `fit(X)` that sets integer
`labels_`, and `predict(X)` that assigns new points to the learned centers. Everything the harness
needs already exists — the data matrix, a Euclidean distance, the arithmetic mean, a way to draw random
points, and the generic "assign, then recompute" alternation used by squared-error local search.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


def _sq_dist_to_nearest(X, centers):
    """For each point, squared Euclidean distance to its nearest center."""
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)   # (n, k)
    return d2.min(axis=1), d2.argmin(axis=1)


class CustomClustering(BaseEstimator, ClusterMixin):
    """Squared-error clustering: place k centers, assign each point to its
    nearest center, minimizing sum_x min_c ||x - c||^2."""

    def __init__(self, n_clusters=None, random_state=42, max_iter=300):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.labels_ = None
        self.cluster_centers_ = None

    def _init_centers(self, X, k, rng):
        # TODO: choose the k initial centers from X.
        #       This placement rule is intentionally blank here.
        pass

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        k = self.n_clusters if self.n_clusters is not None else 8
        centers = self._init_centers(X, k, rng)
        # generic squared-error refine loop: assign to nearest, move to mean
        for _ in range(self.max_iter):
            _, labels = _sq_dist_to_nearest(X, centers)
            new_centers = centers.copy()
            for j in range(k):
                members = X[labels == j]
                if len(members) > 0:
                    new_centers[j] = members.mean(axis=0)
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        self.cluster_centers_ = centers
        self.labels_ = labels
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            self.fit(X)
        _, labels = _sq_dist_to_nearest(X, self.cluster_centers_)
        return labels
```
