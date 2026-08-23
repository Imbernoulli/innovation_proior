class CustomAnomalyDetector:
    """Random-subspace histogram views against irrelevant coordinates.

    Objective of this variant: on wide standardized tables, coordinates with
    no anomaly signal add noise to every full-space distance and every
    pooled tail statistic, so an anomaly deviating in a few coordinates is
    averaged away. Scoring here happens in many low-dimensional views so a
    signal-bearing subset can outvote diluted full-space evidence — but the
    placeholder still picks its views blindly.

    Interface (fixed):
        __init__(self)               -- no required arguments
        fit(self, X) -> self         -- unsupervised, X standardized
        decision_function(self, X)   -- (n_samples,) scores, higher = anomalous

    Adaptation channels:
      - `feature_weight_`: per-feature sampling probability, initialised
        uniform and never updated. Estimating relevance without labels
        (dispersion shape, bimodality, dependence between coordinates) and
        biasing the sampling — or weighting the votes — is the intended
        contribution.
      - `_view_score(view, X)`: per-view score = mean negative log histogram
        density over the view's features (per-feature histograms fitted on
        the train sample once, shared across views).
      - rank pooling: each view's scores are converted to train-referenced
        ranks before averaging, so no single view's scale dominates.
      - `view_dim` never exceeds n_features, so the machinery collapses
        gracefully on narrow data such as thyroid (6 features).

    Available libraries: numpy, scipy, scikit-learn, pyod (import inside
    methods).
    """

    def __init__(self, n_views=24, bins=16, probe_size=4096):
        self.n_views = n_views
        self.bins = bins
        self.probe_size = probe_size
        self.feature_weight_ = None
        self.views_ = []
        self.edges_ = []
        self.logdens_ = []
        self.ref_sorted_ = []

    def _feature_nll(self, j, x):
        """Negative log histogram density of feature j at values x."""
        idx = np.searchsorted(self.edges_[j], x, side="right") - 1
        idx = np.clip(idx, 0, len(self.logdens_[j]) - 1)
        return self.logdens_[j][idx]

    def _view_score(self, view, X):
        s = np.zeros(X.shape[0], dtype=np.float64)
        for j in view:
            s += self._feature_nll(j, X[:, j])
        return s / max(1, len(view))

    def fit(self, X):
        """Fit shared per-feature histograms and draw the random views."""
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.RandomState(SEED)
        n, d = X.shape
        self.feature_weight_ = np.full(d, 1.0 / d)
        self.edges_, self.logdens_ = [], []
        for j in range(d):
            h, e = np.histogram(X[:, j], bins=self.bins)
            width = np.maximum(np.diff(e), 1e-12)
            dens = (h + 1.0) / ((n + self.bins) * width)
            self.edges_.append(e)
            self.logdens_.append(-np.log(dens))
        q = max(2, min(8, d // 3))
        q = min(q, d)
        self.views_ = [
            rng.choice(d, size=q, replace=False, p=self.feature_weight_)
            for _ in range(self.n_views)
        ]
        probe = X[rng.choice(n, size=min(self.probe_size, n), replace=False)]
        self.ref_sorted_ = [np.sort(self._view_score(v, probe)) for v in self.views_]
        return self

    def decision_function(self, X):
        """Mean train-referenced rank across the subspace views."""
        X = np.asarray(X, dtype=np.float64)
        r = np.zeros(X.shape[0], dtype=np.float64)
        for view, ref in zip(self.views_, self.ref_sorted_):
            s = self._view_score(view, X)
            r += np.searchsorted(ref, s, side="left") / max(1, len(ref))
        return r / max(1, len(self.views_))
