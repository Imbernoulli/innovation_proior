**Problem (from step 1).** DBSCAN owned moons (ARI 0.972) but *collapsed* on the 64-dimensional digits
(ARI 0.0003, silhouette −1.0): a single global `eps` cannot survive high-dimensional distance
concentration, so it swept the set into noise. On a geometric-mean aggregate one dead setting is
ruinous. The next rung must be **non-degenerate on every geometry**, even at the cost of shape
flexibility — and the harness hands it the true `n_clusters`, which DBSCAN refused to use.

**Key idea (K-Means).** Partition into exactly `k` groups by minimizing inertia
`phi(C) = Σ_x min_c ||x − c||²`. Exact minimization is NP-hard, so alternate two forced steps that each
weakly decrease `phi`: assign each point to its nearest center (Voronoi), then move each center to its
group's mean (the unique squared-error minimizer, by the parallel-axis identity). The loop converges to
a local minimum in finitely many sweeps — with **no global radius to mis-set**, so it cannot collapse to
one cluster the way DBSCAN's `eps` did.

**Why it works here.** It returns exactly `k = 10` honest centroids on digits (off the −1.0 floor) and
models the convex isotropic blobs exactly. Because the local minimum is hostage to initialization,
**k-means++** seeds each center with probability ∝ `D(x)²` (squared distance to the nearest chosen
center) — matching the squared-error objective, robust to outliers, with an `E[phi] ≤ 8(ln k + 2)·phi_OPT`
guarantee — and `n_init` restarts keep the lowest-inertia run.

**Hyperparameters / the edit.** `sklearn.cluster.KMeans` *is* this method (Lloyd + `init="k-means++"`
default). Instantiate `KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)` with
`k = n_clusters` from the harness (8 only as an unused fallback). `n_init = 10` restarts; `max_iter = 300`
caps the loop. `predict` delegates to the fitted model's nearest-center rule.

**What to watch.** Digits should jump decisively (silhouette → positive, ARI → real); blobs should beat
DBSCAN (~0.85 ARI). The knowing cost is **moons**: nearest-centroid is a Voronoi tessellation that must
cut a half-moon with a straight bisector, so moons ARI should *fall* below DBSCAN's 0.972 — that
sacrificed non-convex setting is the next rung's target.

```python
class CustomClustering(BaseEstimator, ClusterMixin):
    """K-Means clustering (Lloyd's algorithm)."""

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        self._model = None

    def fit(self, X):
        from sklearn.cluster import KMeans

        k = self.n_clusters if self.n_clusters is not None else 8
        self._model = KMeans(
            n_clusters=k, random_state=self.random_state, n_init=10, max_iter=300
        )
        self._model.fit(X)
        self.labels_ = self._model.labels_
        return self

    def predict(self, X):
        if self._model is None:
            self.fit(X)
        return self._model.predict(X)


def custom_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))
```
