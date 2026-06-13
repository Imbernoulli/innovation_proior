## Research question

Partition unlabeled points into groups that reflect their underlying structure, across three very
different geometries at once: convex isotropic blobs, an interleaving non-convex pair (moons), and a
high-dimensional real embedding (the 8×8 digit images, 64 features). No single classical method
dominates that whole range, and the contribution being designed is the **algorithm itself** — the
assignment rule, the graph/density construction, the initialization, the cluster-extraction rule — not
per-dataset tuning. The single thing each rung edits is one clustering estimator; everything around it
(data generation, standardization, scoring) is fixed.

## Prior art before the first rung (the clustering lineage)

The first rung reacts to the standard toolbox. These are the methods that precede the ladder; the
weakest rung is chosen against their gaps.

- **Partitioning — k-means / Lloyd (Lloyd 1957; MacQueen 1967) and k-medoid / PAM, CLARANS
  (Kaufman & Rousseeuw; Ng & Han 1994).** Pick `k` representatives, send each point to the nearest one,
  minimize within-group cost `E = Σ_j Σ_{p∈C_j} dist(p, rep_j)`; alternate assign/recompute. "Nearest
  representative" *is* a Voronoi tessellation, so every producible cluster is **convex** — a curved band
  or a cluster wrapped around another cannot be a Voronoi cell. Gaps: must be told `k`; convex-only
  (fails non-convex shapes); forces every point into a cluster (no noise); outlier-sensitive.
- **Hierarchical / connectivity (agglomerative + a cut threshold; Ejcluster, García et al.).**
  Merge-or-split into a dendrogram, no `k` up front. "Walk in small steps" connectivity (Ejcluster)
  *is* shape-agnostic and handles non-convex groups. Gaps: needs an unsettable termination threshold
  `D_min` (too small fragments a loose group, too large fuses separate ones, and with clusters of
  different tightness no single value does both); the connectivity variant costs `O(n²)`.
- **Grid / histogram density (Jain's line).** Bin the space, call high-count cells cores, put boundaries
  in the histogram valleys — reads density off the data and can find arbitrary shapes. Gaps: storage and
  search blow up with dimension; everything hinges on a guessed cell size (too coarse merges, too fine
  fragments).

The recurring signal is **density** — groups are dense, gaps and noise are sparse — but partitioning is
convex-only and noise-blind, while the connectivity/grid ideas that get shape right are either
quadratic or hinge on an unsettable threshold. The first rung is the density-based method that resolves
exactly that tension.

## The fixed substrate

A single benchmark harness is frozen and must not be touched. It generates each dataset, applies a
`StandardScaler` (so every method sees zero-mean/unit-variance features — this is load-bearing for any
radius-based method), instantiates the editable estimator with the true cluster count passed as
`n_clusters` and the seed as `random_state`, calls `fit(X)` then `predict(X)`, and scores the returned
labels against the held-out ground truth. The harness imports for the editable region: `numpy`,
`sklearn.base.BaseEstimator`, `sklearn.base.ClusterMixin`, `sklearn.preprocessing.StandardScaler`,
`sklearn.metrics.*`; any module from `scikit-learn`, `numpy`, or `scipy` may be imported inside the
edited code. The silhouette term needs `2 ≤ #labels < n`; a labeling that collapses to one cluster (or
all-noise) is scored `silhouette = −1.0`, so degenerate outputs are penalized hard.

## The editable interface

Exactly one region of `scikit-learn/custom_clustering.py` is editable (lines 36–109): the
`CustomClustering` estimator and the optional `custom_distance` helper. Every rung is a fill of this
same contract — `__init__(self, n_clusters=None, random_state=42)`, `fit(X) -> self` (sets
`self.labels_`), `predict(X) -> labels`. The harness passes the true cluster count as `n_clusters`, but
a method that determines its own cluster count may ignore it.

The starting point is the scaffold default: a plain K-Means fallback (it always returns *some* labels,
so the harness never crashes — it is the floor to beat, not a serious method). Each rung replaces
exactly this region.

```python
# EDITABLE region of custom_clustering.py (lines 36-109) -- default fill (K-Means fallback)
class CustomClustering(BaseEstimator, ClusterMixin):
    """Custom clustering algorithm.

    fit(X) -> self        : fit to data X (n_samples, n_features); sets self.labels_
    predict(X) -> labels  : cluster labels for X (n_samples,)
    """

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X):
        # Default: simple K-Means fallback.
        from sklearn.cluster import KMeans

        k = self.n_clusters if self.n_clusters is not None else 8
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        km.fit(X)
        self.labels_ = km.labels_
        return self

    def predict(self, X):
        # Default: refit (stateless fallback).
        self.fit(X)
        return self.labels_


def custom_distance(x, y):
    """Custom distance metric between two points (1-D arrays). Returns float >= 0."""
    return np.sqrt(np.sum((x - y) ** 2))
```

## Evaluation settings

Three datasets span the geometry range, each over three seeds {42, 123, 456}:

- **blobs** — 1500 points, 5 isotropic Gaussian clusters with *varying* spread
  (`cluster_std = [0.8, 1.2, 0.5, 1.5, 1.0]`), then StandardScaled. Convex but multi-density.
- **moons** — 1000 points, two interleaving half-circles, `noise = 0.08`. Non-convex.
- **digits** — `load_digits()`, 1797 points, 10 classes, 64 features. High-dimensional real data.

Three metrics, **higher is better on all**: **ARI** (Adjusted Rand Index vs ground truth), **NMI**
(Normalized Mutual Information vs ground truth), **Silhouette** (intrinsic compactness/separation, no
labels). Each dataset's score is the mean of its three metrics; the task aggregate is the geometric
mean across the three datasets — so a single collapsed setting (silhouette floored at −1.0) drags the
whole task down, and a method must be at least non-degenerate on *every* geometry to score.
