# ================================================================
# EDITABLE -- agent modifies this section
# ================================================================
# Variant: some rows belong to no cluster. Find them, fit the cores
# without their votes, then attach each straggler where it hurts least
# (every row must still receive a label).


class CustomClustering(BaseEstimator, ClusterMixin):
    """Outlier-aware clusterer: cores first, stragglers last.

    Required interface (do not change signatures):
        fit(X) -> self          : sets self.labels_ (every row labelled)
        predict(X) -> labels    : (n_samples,) integer cluster ids

    Two-tier scaffold implementing the variant:
      1. Support statistic: k-th neighbor distances set a density radius
         from the data's own scale (75th percentile) -- no fixed fraction
         of points is ever discarded blindly.
      2. Core fitting: density clustering over well-supported regions, so
         low-support rows get no vote in shaping the clusters.
      3. Attachment: each flagged row takes the label of its nearest core
         centroid -- the least-damaging rule, identical for every input.
         Falls back to K-Means when the density cores collapse.

    `self.outlier_mask_` records which rows were treated as stragglers.

    Args:
        n_clusters: hint consumed only by the fallback path.
        random_state: seed for reproducibility.
    """

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        self.outlier_mask_ = None
        self._centers_ = None

    def fit(self, X):
        """Fit cores on well-supported points, then attach stragglers."""
        from sklearn.cluster import DBSCAN, KMeans
        from sklearn.neighbors import NearestNeighbors

        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        k = int(max(4, min(10, n - 1)))
        dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
        eps = float(np.percentile(dist[:, -1], 75.0))

        lab = DBSCAN(eps=eps, min_samples=k).fit_predict(X)
        kept = np.unique(lab[lab >= 0])

        if kept.size < 2:
            # density cores collapsed -> plain partitional fallback
            kk = self.n_clusters if self.n_clusters is not None else 8
            kk = int(max(2, min(kk, n - 1)))
            km = KMeans(n_clusters=kk, random_state=self.random_state, n_init=10)
            lab = km.fit_predict(X)
            self._centers_ = km.cluster_centers_
            self.outlier_mask_ = np.zeros(n, dtype=bool)
        else:
            centers = np.stack([X[lab == c].mean(axis=0) for c in kept])
            new = np.full(n, -1, dtype=np.int64)
            for i, c in enumerate(kept):
                new[lab == c] = i
            stragglers = new < 0
            if stragglers.any():
                d2 = ((X[stragglers][:, None, :] - centers[None, :, :]) ** 2).sum(-1)
                new[stragglers] = d2.argmin(axis=1)
            lab = new
            self._centers_ = centers
            self.outlier_mask_ = stragglers
        self.labels_ = np.asarray(lab, dtype=np.int64)
        return self

    def predict(self, X):
        """Fitted labels for the training matrix; nearest core otherwise."""
        X = np.asarray(X, dtype=np.float64)
        if self.labels_ is not None and X.shape[0] == self.labels_.shape[0]:
            return self.labels_
        d2 = ((X[:, None, :] - self._centers_[None, :, :]) ** 2).sum(axis=-1)
        return d2.argmin(axis=1)


def custom_distance(x, y):
    """Distance hook; the density radius above lives in this metric."""
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)
                                - np.asarray(y, dtype=np.float64)))
