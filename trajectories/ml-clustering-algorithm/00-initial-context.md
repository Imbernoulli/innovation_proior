## Research question

Partition unlabeled points into groups that reflect their underlying structure across three geometries at once: convex isotropic blobs, an interleaving non-convex pair (moons), and a high-dimensional real embedding (the 8×8 digit images, 64 features). The contribution being designed is the clustering algorithm itself — the assignment rule, the graph/density construction, the initialization, and the cluster-extraction rule — not per-dataset tuning. Only the estimator is editable; data generation, standardization, and scoring are fixed.

## Prior art / Background / Baselines

- **Partitioning — k-means / Lloyd, k-medoids / PAM, CLARANS.** These methods pick a fixed number of representatives and assign each point to the nearest one, alternating assignment and recomputation to minimize within-group cost. They must be told `k`; the nearest-representative rule produces convex Voronoi cells.

- **Hierarchical / connectivity — agglomerative clustering with a cut threshold and connectivity variants.** They merge or split points into a dendrogram, so they need no `k` up front and can follow non-convex connectivity. They require a termination threshold to determine the final partition; the connectivity variant scales as `O(n²)`.

- **Grid / histogram density.** They bin the space, treat high-count cells as cores, and place boundaries in histogram valleys, allowing arbitrary cluster shapes.

## Fixed substrate / Code framework

A single benchmark harness is frozen. It generates each dataset, applies `StandardScaler`, instantiates the editable estimator with the true cluster count as `n_clusters` and the seed as `random_state`, calls `fit(X)` then `predict(X)`, and scores the returned labels against held-out ground truth. Allowed imports for the editable region include `numpy`, `sklearn.base.BaseEstimator`, `sklearn.base.ClusterMixin`, `sklearn.preprocessing.StandardScaler`, `sklearn.metrics.*`, and any module from `scikit-learn`, `numpy`, or `scipy`. The silhouette term requires `2 ≤ #labels < n`; a degenerate single-cluster or all-noise labeling scores `silhouette = −1.0`.

## Editable interface

Exactly one region of `scikit-learn/custom_clustering.py` is editable: the `CustomClustering` estimator and the optional `custom_distance` helper. The contract is `__init__(self, n_clusters=None, random_state=42)`, `fit(X) -> self` (sets `self.labels_`), `predict(X) -> labels`. The harness passes the true cluster count as `n_clusters`, but a method that determines its own cluster count may ignore it.

The default scaffold is a plain K-Means fallback. Each rung replaces this region.

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

Three datasets span the geometry range, each over seeds {42, 123, 456}:

- **blobs** — 1500 points, 5 isotropic Gaussian clusters with varying spread (`cluster_std = [0.8, 1.2, 0.5, 1.5, 1.0]`), then StandardScaled. Convex but multi-density.
- **moons** — 1000 points, two interleaving half-circles, `noise = 0.08`. Non-convex.
- **digits** — `load_digits()`, 1797 points, 10 classes, 64 features. High-dimensional real data.

Three metrics, higher is better: **ARI** (Adjusted Rand Index vs ground truth), **NMI** (Normalized Mutual Information vs ground truth), and **Silhouette** (intrinsic compactness/separation). Each dataset's score is the mean of its three metrics; the task aggregate is the geometric mean across the three datasets. A collapsed setting, with silhouette floored at −1.0, drags the whole task down, so a method must be non-degenerate on every geometry.
