# LOF (Local Outlier Factor), distilled

LOF assigns each point a real-valued *degree of being an outlier*, measured **locally**: how
much sparser a point's surroundings are than the surroundings of its own nearest neighbors.
It is simultaneously distribution-free, defined for any metric
and any dimensionality, real-valued (rankable rather than binary), and — crucially — *local*,
so it works on data containing clusters of very different densities. A point deep inside any
cluster scores ≈ 1; a point in a sparse pocket whose neighbors are dense scores well above 1.

## Problem it solves

Unsupervised outlier scoring on multidimensional data whose regions have **different
densities**. Global distance-based detectors (`DB(p, D)`-outliers; rank-by-`Dᵏ` outliers)
compare every point to a single distance scale, so they cannot separate a genuine anomaly next
to a dense cluster from an ordinary member of a sparse cluster — the absolute distances coincide.
LOF replaces the global scale with a per-point comparison to the *local* neighborhood density.

## Key idea and definitions

Fix a neighborhood size `MinPts` (= `k` = `n_neighbors`). For point `p`:

- **k-distance(p):** distance from `p` to its `MinPts`-th nearest neighbor.
- **k-distance neighborhood `N(p)`:** all objects within `k-distance(p)` (may exceed `MinPts`
  under ties — always use `|N(p)|` in the exact definition; fixed-width implementations resolve
  ties by the neighbor table they materialize).
- **Reachability distance** (smooths the density estimate; floored by the *neighbor's*
  k-distance, hence asymmetric):

  ```
  reach-dist(p, o) = max{ k-distance(o), d(p, o) }
  ```

  Flooring every `p` inside `o`'s neighborhood to `k-distance(o)` collapses local fluctuations;
  larger `MinPts` ⇒ stronger smoothing.

- **Local reachability density** (inverse of the *average* reach-distance, so `|N(p)|` cancels
  and densities are comparable across points):

  ```
  lrd(p) = 1 / ( ( Σ_{o ∈ N(p)} reach-dist(p, o) ) / |N(p)| )
  ```

- **Local outlier factor** (the ratio makes it local: absolute density cancels):

  ```
  LOF(p) = ( Σ_{o ∈ N(p)} lrd(o) / lrd(p) ) / |N(p)|
         = ( average lrd over p's neighbors ) / ( lrd(p) )
  ```

Interpretation: `LOF ≈ 1` — as dense as your neighbors (inlier); `LOF < 1` — denser (inlier);
`LOF > 1` — emptier surroundings than your neighbors (outlier), larger = stronger.

## Why it behaves (the guarantees)

**Deep-point lemma — interior points score ≈ 1.** For a cluster `C` with min/max within-cluster
reach-distances `reach-dist-min`/`reach-dist-max` and `ε = reach-dist-max/reach-dist-min − 1`,
any `p` whose neighbors and neighbors-of-neighbors all lie in `C` satisfies

```
1/(1+ε) ≤ LOF(p) ≤ (1+ε).
```

Proof: every reach-distance in `lrd(p)` lies in `[reach-dist-min, reach-dist-max]`, so
`1/reach-dist-max ≤ lrd(p) ≤ 1/reach-dist-min`, and the same for each neighbor; the ratios
`lrd(o)/lrd(p)` are thus in `[reach-dist-min/reach-dist-max, reach-dist-max/reach-dist-min]`, and
their average stays there. Tighter cluster (smaller `ε`) ⇒ closer to exactly 1. This is what
makes interior points of *any* cluster — dense or sparse — score the same "ordinary" value.

**General bound — any point.** With `direct_min/max(p)` = min/max reach-distance over `N(p)`, and
`indirect_min/max(p)` = min/max reach-distance over the neighbors' neighborhoods,

```
direct_min(p)/indirect_max(p) ≤ LOF(p) ≤ direct_max(p)/indirect_min(p).
```

(Lower bound: `lrd(p) ≤ 1/direct_min`, `lrd(o) ≥ 1/indirect_max`, substitute into the average;
upper bound is the mirror.) `LOF` is governed by the ratio of `p`'s reach-distances to its
neighbors'.

**Tightness.** If reach-distances fluctuate by `pct` around their means, the relative spread is

```
(LOF_max − LOF_min) / (direct/indirect) = (4·pct/100) / (1 − (pct/100)²),
```

which depends **only on `pct`**, not on absolute distances — the local property quantified.
Homogeneous neighborhood (small `pct`) ⇒ sharp `LOF`; the bound goes slack only when `p`'s
neighbors straddle clusters of differing density (`pct → 100`).

**Multi-cluster bound.** Partition `N(p)` into groups `C₁..C_n` with weights `ξᵢ = |Cᵢ|/|N(p)|`:

```
LOF(p) ≥ ( Σᵢ ξᵢ·direct^i_min ) · ( Σᵢ ξᵢ/indirect^i_max ),
LOF(p) ≤ ( Σᵢ ξᵢ·direct^i_max ) · ( Σᵢ ξᵢ/indirect^i_min ).
```

These come from bounding `1/lrd(p)`, the average reach-distance for `p`, group by group, and
bounding the average neighbor `lrd` group by group; multiplying the two positive factors gives the
LOF bounds.
One group (`n=1`, `ξ₁=1`) reduces exactly to the general bound.

## Choosing MinPts

`LOF` is **not monotonic** in `MinPts` (it fluctuates even on a pure Gaussian blob). Practical
recipe: pick a range `[MinPtsLB, MinPtsUB]`, compute `LOF` at each value, and score each point by
the **maximum** `LOF` over the range (max highlights the scale at which the point is most
outlying; min erases it, mean dilutes it).

- `MinPtsLB ≥ ~10`: below ~10 the std-dev of `LOF` is still large (statistical fluctuation
  dominates). Also reads as the minimum cluster size you want points to be outliers relative to.
- `MinPtsUB`: the largest group of "close by" points you'd still want to flag as local outliers
  (domain-dependent); 10–20 works well as a default range.

A single fixed `MinPts = 20` is the common default (`n_neighbors=20`). Degenerate case: `≥ MinPts`
exact duplicates make the average reach-distance 0 (`lrd → ∞`); floor the denominator with a tiny
`ε` (e.g. `1e-10`).

## Working code

A compact implementation faithful to the `sklearn.neighbors.LocalOutlierFactor` core: materialize
fit-time neighbors, compute
reach-distance → `lrd` → `LOF`, store `negative_outlier_factor_ = -LOF`, and expose both the
standard opposite-LOF scores and a positive anomaly-score convenience. `n_neighbors=20`,
Minkowski `p=2` gives Euclidean distance by default.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors


class LOF:
    """Local-density ratio with the standard opposite-LOF scoring convention."""

    def __init__(
        self,
        n_neighbors=20,
        *,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        contamination="auto",
        n_jobs=None,
    ):
        self.n_neighbors = n_neighbors      # MinPts: neighborhood size / smoothing
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.contamination = contamination
        self.n_jobs = n_jobs

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))

        # +1 because a training point is its own 0-distance neighbor; drop that column below.
        self._nn = NearestNeighbors(
            n_neighbors=self.n_neighbors_ + 1,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        ).fit(X)
        distances, neighbors = self._nn.kneighbors(X)
        self._distances_fit_X_ = distances[:, 1:]
        neighbors = neighbors[:, 1:]            # exclude self

        self._lrd = self._local_reachability_density(self._distances_fit_X_, neighbors)
        lrd_ratios = self._lrd[neighbors] / self._lrd[:, None]
        self.negative_outlier_factor_ = -np.mean(lrd_ratios, axis=1)

        if self.contamination == "auto":
            self.offset_ = -1.5
        else:
            self.offset_ = np.percentile(
                self.negative_outlier_factor_,
                100.0 * self.contamination,
            )
        self.decision_scores_ = -self.negative_outlier_factor_  # higher = more anomalous
        return self

    def _local_reachability_density(self, distances_X, neighbors_indices):
        # reach-dist(p, o) = max{ k-distance(o), d(p, o) }  -- floor by NEIGHBOR's k-distance.
        dist_k = self._distances_fit_X_[neighbors_indices, self.n_neighbors_ - 1]
        reach = np.maximum(distances_X, dist_k)
        # lrd = 1 / mean reach-dist; eps floors the >=MinPts-duplicates case (avg reach = 0).
        return 1.0 / (reach.mean(axis=1) + 1e-10)

    def score_samples(self, X):
        """Opposite LOF for new query points; larger values mean more normal."""
        X = np.asarray(X, dtype=float)
        distances_X, neighbors_X = self._nn.kneighbors(
            X,
            n_neighbors=self.n_neighbors_,
        )
        X_lrd = self._local_reachability_density(distances_X, neighbors_X)
        lrd_ratios = self._lrd[neighbors_X] / X_lrd[:, None]
        return -np.mean(lrd_ratios, axis=1)

    def decision_function(self, X):
        """Shifted opposite LOF; zero is the inlier/outlier threshold."""
        return self.score_samples(X) - self.offset_

    def anomaly_score(self, X):
        """Positive LOF-style score for APIs where higher means more anomalous."""
        return -self.score_samples(X)
```

The local `sklearn.neighbors.LocalOutlierFactor` snapshot uses the same core:
`dist_k = fitted_distances[neighbor_indices, n_neighbors_ - 1]`,
`reach_dist = max(query_dist, dist_k)`, `lrd = 1/(mean(reach_dist)+1e-10)`, and
`negative_outlier_factor_ = -mean(lrd[neighbors]/lrd_self)`. `score_samples` returns the
opposite LOF for query points (larger = more normal), and `decision_function` subtracts the
`offset_` (default `-1.5`) so that threshold `0` separates inliers from outliers. Wrappers with a
"higher = more anomalous" contract invert the opposite-LOF score.
