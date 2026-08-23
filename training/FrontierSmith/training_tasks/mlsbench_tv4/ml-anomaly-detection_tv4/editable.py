class CustomAnomalyDetector:
    """Complementary-family basis with a worst-type-aware aggregation hook.

    Objective of this variant: anomalies come in types — marginal tail
    points, correlation breakers, tight fault clusters — and each scorer
    family is blind to some of them. Three deliberately complementary
    scorers are fitted; per-dataset performance is capped by whichever type
    the aggregation lets slip, and the placeholder mean is exactly the
    aggregation that lets a minority type slip.

    Interface (fixed):
        __init__(self)               -- no required arguments
        fit(self, X) -> self         -- unsupervised, X standardized
        decision_function(self, X)   -- (n_samples,) scores, higher = anomalous

    The basis (each with a distinct blind spot):
      - `_tail_score`: per-dimension empirical tail mass — sees marginal
        outliers, blind to broken correlations.
      - `_maha_score`: Mahalanobis distance under a ridge-regularised
        covariance — sees broken correlations, blind to local structure.
      - `_local_score`: mean k-NN distance to a reference subsample — sees
        local/cluster anomalies, blind to smooth global tails.

    Adaptation channels:
      - `_aggregate(Z)`: takes the (n, 3) matrix of per-family z-scores;
        placeholder = plain row mean, which dilutes any single family's
        evidence by 3x. Worst-case-aware rules (max, soft-max weighting,
        per-point family selection) are the intended replacement.
      - `mu_` / `sd_`: per-family standardisation from a train probe — what
        makes the families commensurable before aggregation.

    Available libraries: numpy, scipy, scikit-learn, pyod (import inside
    methods).
    """

    def __init__(self, ref_size=2048, k=10, ridge=1e-3, probe_size=4096):
        self.ref_size = ref_size
        self.k = k
        self.ridge = ridge
        self.probe_size = probe_size
        self.cols_ = None
        self.mean_ = None
        self.prec_ = None
        self.nn_ = None
        self.mu_ = None
        self.sd_ = None

    def _tail_score(self, X):
        n_ref = self.cols_[0].shape[0]
        s = np.zeros(X.shape[0], dtype=np.float64)
        for j, col in enumerate(self.cols_):
            r = np.searchsorted(col, X[:, j], side="left") / max(1, n_ref)
            r = np.clip(r, 1e-6, 1.0 - 1e-6)
            s += -np.log(2.0 * np.minimum(r, 1.0 - r))
        return s / len(self.cols_)

    def _maha_score(self, X):
        D = X - self.mean_
        q = np.einsum("ij,jk,ik->i", D, self.prec_, D)
        return np.sqrt(np.maximum(q, 0.0))

    def _local_score(self, X):
        d, _ = self.nn_.kneighbors(X)
        return d.mean(axis=1)

    def _family_matrix(self, X):
        return np.column_stack(
            [self._tail_score(X), self._maha_score(X), self._local_score(X)]
        )

    def _aggregate(self, Z):
        """Placeholder aggregation: plain mean over the three families."""
        return Z.mean(axis=1)

    def fit(self, X):
        """Fit all three families and their common standardisation."""
        from sklearn.neighbors import NearestNeighbors

        X = np.asarray(X, dtype=np.float64)
        rng = np.random.RandomState(SEED)
        n, d = X.shape
        self.cols_ = [np.sort(X[:, j]) for j in range(d)]
        self.mean_ = X.mean(axis=0)
        cov = np.cov(X, rowvar=False) + self.ridge * np.eye(d)
        self.prec_ = np.linalg.pinv(cov)
        ref = X[rng.choice(n, size=min(self.ref_size, n), replace=False)]
        self.nn_ = NearestNeighbors(n_neighbors=min(self.k, len(ref))).fit(ref)
        probe = X[rng.choice(n, size=min(self.probe_size, n), replace=False)]
        F = self._family_matrix(probe)
        self.mu_ = F.mean(axis=0)
        self.sd_ = F.std(axis=0) + 1e-12
        return self

    def decision_function(self, X):
        """Aggregate standardised family scores (higher = more anomalous)."""
        X = np.asarray(X, dtype=np.float64)
        Z = (self._family_matrix(X) - self.mu_) / self.sd_
        return self._aggregate(Z)
