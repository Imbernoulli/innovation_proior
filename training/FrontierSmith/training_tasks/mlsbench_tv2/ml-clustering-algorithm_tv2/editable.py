# ================================================================
# EDITABLE -- agent modifies this section
# ================================================================
# Variant: the cluster count is untrusted metadata. The fitted number of
# groups must come from a selection rule computed on X alone.


class CustomClustering(BaseEstimator, ClusterMixin):
    """Count-free clusterer: discovers how many groups the data holds.

    Required interface (do not change signatures):
        fit(X) -> self          : sets self.labels_ for X (n_samples, n_features)
        predict(X) -> labels    : (n_samples,) integer cluster ids

    Variant rules baked into this scaffold:
    - `n_clusters` is stored as `self.k_hint_` for reference only; no code
      path may let it decide the fitted count `self.k_`.
    - `_choose_k` is the model-selection channel: one criterion scanned
      over candidate counts, applied identically to every input. The
      placeholder sweeps an internal quality score; replace it with a
      stronger rule (stability selection, gap statistic, eigengap,
      density-mode counting).

    Args:
        n_clusters: metadata hint, recorded but never obeyed.
        random_state: seed for reproducibility.
    """

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        self.k_hint_ = None
        self.k_ = None
        self._centers_ = None

    def _choose_k(self, X, k_min=2, k_max=12):
        """Pick the cluster count from the data (placeholder: quality sweep).

        Fits a cheap K-Means per candidate count and keeps the silhouette
        maximiser. Crude -- this sweep is the part to outgrow.
        """
        from sklearn.cluster import KMeans

        n = X.shape[0]
        best_k, best_s = k_min, -np.inf
        for k in range(k_min, int(min(k_max, n - 1)) + 1):
            lab = KMeans(n_clusters=k, random_state=self.random_state,
                         n_init=4).fit_predict(X)
            try:
                s = silhouette_score(X, lab)
            except ValueError:
                continue
            if s > best_s:
                best_k, best_s = k, s
        return best_k

    def fit(self, X):
        """Cluster X with a self-determined number of groups."""
        from sklearn.cluster import KMeans

        X = np.asarray(X, dtype=np.float64)
        self.k_hint_ = self.n_clusters          # reference only, never obeyed
        self.k_ = self._choose_k(X)
        km = KMeans(n_clusters=self.k_, random_state=self.random_state, n_init=10)
        self.labels_ = km.fit_predict(X)
        self._centers_ = km.cluster_centers_
        return self

    def predict(self, X):
        """Fitted labels for the training matrix; nearest center otherwise."""
        X = np.asarray(X, dtype=np.float64)
        if self.labels_ is not None and X.shape[0] == self.labels_.shape[0]:
            return self.labels_
        d2 = ((X[:, None, :] - self._centers_[None, :, :]) ** 2).sum(axis=-1)
        return d2.argmin(axis=1)


def custom_distance(x, y):
    """Distance hook available to the algorithm (plain euclidean for now)."""
    return float(np.sqrt(np.sum((x - y) ** 2)))
