**Problem (from steps 1–2).** DBSCAN owns non-convex moons (ARI 0.972) but dies on high-D digits
(silhouette −1.0); K-Means owns convex blobs and digits but dies on moons (ARI 0.481). Each is
excellent-on-some, terrible-on-another — the wrong profile for a geometric-mean aggregate. The fix must
keep DBSCAN's arbitrary shape and explicit noise *without* its fatal weakness: a single global `eps`
sets one density level, and the multi-density blobs and high-D digits live at many densities.

**Key idea (HDBSCAN).** Don't pick one `eps` — build the whole density hierarchy and read a flat
clustering off it at *different* levels. Replace distances by **mutual reachability**
`d_mreach(p,q) = max(d_core(p), d_core(q), d(p,q))` (inflates sparse bridges, damping single-linkage
chaining), run **single-linkage** on its MST (= DBSCAN at every radius at once), **condense** the tree
by a minimum cluster size (true split vs shrink vs die), and extract the flat clustering that maximizes
total **relative excess of mass** (EOM) via a bottom-up DP. Diffuse and dense clusters can both appear,
selected at different levels — the exact thing one global `eps` cannot do.

**Why it works here.** Self-tuning multi-level density: it finds the moons arcs without a hand-set `eps`,
handles the varying-`cluster_std` blobs by selecting per-level, and determines its own cluster count
(the self-determination DBSCAN had, K-Means lacked).

**Hyperparameters / the edit.** `sklearn.cluster.HDBSCAN` realizes the construction. Set
`min_cluster_size = max(5, n // 50)` (granularity scaled to dataset size), `min_samples = 5` (density
smoothing), `cluster_selection_method = "eom"`. It ignores the harness's `n_clusters` by design. One
harness guard from step 2's lesson: if HDBSCAN labels *everything* noise (`len(set(labels)) ≤ 1`, the
−1.0-silhouette collapse), fall back to `KMeans(n_clusters)` so the harness never gets a degenerate
labeling.

**What to watch.** Moons should match/exceed DBSCAN (ARI ≈ 1.0); blobs should beat DBSCAN's capped 0.70;
digits should be non-degenerate but modest (likely below K-Means's 0.534, since 64-D density estimation
leaves many points as noise). The bet: HDBSCAN is the only rung non-degenerate on *all three* geometries
at once, so its geometric-mean aggregate tops both prior rungs.

```python
class CustomClustering(BaseEstimator, ClusterMixin):
    """HDBSCAN — hierarchical density-based clustering (Campello et al., 2013)."""

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X):
        from sklearn.cluster import HDBSCAN

        # HDBSCAN automatically determines the number of clusters.
        # min_cluster_size controls granularity.
        min_cs = max(5, X.shape[0] // 50)
        self._model = HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=5,
            cluster_selection_method="eom",
        )
        self._model.fit(X)
        self.labels_ = self._model.labels_

        # If HDBSCAN assigns everything to noise (-1), fall back to
        # labeling all points as cluster 0 to avoid degenerate metrics.
        if len(set(self.labels_)) <= 1:
            from sklearn.cluster import KMeans
            k = self.n_clusters if self.n_clusters is not None else 8
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            km.fit(X)
            self.labels_ = km.labels_

        return self

    def predict(self, X):
        if self.labels_ is None:
            self.fit(X)
        return self.labels_


def custom_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))
```
