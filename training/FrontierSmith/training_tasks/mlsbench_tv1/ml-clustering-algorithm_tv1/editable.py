# ================================================================
# EDITABLE -- agent modifies this section
# ================================================================
# Target: the WEAKEST setting, under ONE fixed configuration.
# A setting only counts as solved when label agreement and geometric
# compactness are both respectable; a large gap between them means the
# metric/representation, not the assignment rule, is the bottleneck.


class CustomClustering(BaseEstimator, ClusterMixin):
    """Single-configuration clusterer judged by its weakest input.

    Required interface (do not change these signatures):
        fit(X) -> self          : X is (n_samples, n_features); must set self.labels_
        predict(X) -> labels    : (n_samples,) integer cluster ids

    Design targets for this variant:
    - The SAME instance and hyperparameters see every input. Adapt through a
      data statistic computed inside fit (see `_scale_statistic`), used as a
      continuous quantity — not an if/else that dispatches to a named method.
    - Close the gap between extrinsic agreement and intrinsic compactness. If
      assignments are right but the clusters are diffuse, fix the geometry
      (metric learning, local scaling, affinity re-weighting, an embedding you
      fit yourself) rather than relabelling.
    - Treat `n_clusters` as a hint that may be absent or wrong.

    Args:
        n_clusters: optional metadata hint for the number of clusters.
        random_state: seed for reproducibility.
    """

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        # Populated by fit(); available to predict() and custom_distance().
        self.scale_ = None

    def _scale_statistic(self, X):
        """A scale-free description of the local geometry of X.

        The placeholder returns the median distance to the k-th nearest
        neighbour, which is small for tight blobs, large for diffuse
        high-dimensional data, and does not depend on any dataset identity.
        Replace or extend it — this is the intended adaptation channel.
        """
        n = X.shape[0]
        k = max(2, min(10, n - 1))
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n, size=min(n, 256), replace=False)
        d = np.sqrt(((X[idx, None, :] - X[None, :, :]) ** 2).sum(-1))
        d.sort(axis=1)
        return float(np.median(d[:, k]))

    def fit(self, X):
        """Fit the clusterer to X and set self.labels_.

        Args:
            X: array of shape (n_samples, n_features)

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        self.scale_ = self._scale_statistic(X)

        # Placeholder: one fixed configuration, no per-dataset branching. It is
        # deliberately weak on the geometry side -- that gap is the task.
        from sklearn.cluster import KMeans

        k = self.n_clusters if self.n_clusters is not None else 8
        k = int(max(2, min(k, max(2, X.shape[0] // 2))))
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        km.fit(X)
        self.labels_ = km.labels_
        return self

    def predict(self, X):
        """Predict cluster labels for X.

        Args:
            X: array of shape (n_samples, n_features)

        Returns:
            labels: array of shape (n_samples,)
        """
        self.fit(X)
        return self.labels_


def custom_distance(x, y, scale=None):
    """Distance used by your method — the main lever for the geometry gap.

    Args:
        x, y: 1-D arrays of shape (n_features,)
        scale: optional local-scale statistic (see CustomClustering.scale_)

    Returns:
        distance: float >= 0
    """
    d = np.sqrt(np.sum((x - y) ** 2))
    return d if not scale else d / scale
