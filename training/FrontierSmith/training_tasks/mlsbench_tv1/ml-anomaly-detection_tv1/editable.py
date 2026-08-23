class CustomAnomalyDetector:
    """Contamination-regime-robust anomaly scoring.

    Objective of this variant: one detector, one fixed configuration, that
    stays strong when the anomaly rate is anywhere between ~2.5% and ~31.6% of
    the data — adapting only through statistics computed from the unlabeled
    features it is fitted on. Ranking quality (AUROC) and post-threshold
    decision quality (F1) are weighted equally, so the SHAPE of the score
    distribution near the operating point matters as much as its ordering.

    Interface (fixed):
        __init__(self)               -- no required arguments
        fit(self, X) -> self         -- unsupervised, X standardized
        decision_function(self, X)   -- (n_samples,) scores, higher = anomalous

    Structure provided as the intended adaptation channels:
      - a subsampled multi-view ensemble (`n_views` × `subsample`): each view
        scores points by mean distance to its k nearest reference points;
        views decorrelate the estimate on large / high-contamination data.
      - `_contamination_signal(scores)`: estimate of the anomaly fraction
        derived from an unlabeled score sample. The placeholder returns a
        constant 0.1 — replacing it with a real statistic (score-gap, mixture
        fit, tail mass) and using it to reshape scores is the point of the
        variant.
      - `scale_`: median train score, used to put scores on a comparable
        scale across datasets.

    Available libraries: numpy, scipy, scikit-learn, pyod (import inside
    methods).
    """

    def __init__(self, n_views=4, subsample=256, k=10):
        self.n_views = n_views
        self.subsample = subsample
        self.k = k
        self.views_ = []
        self.scale_ = 1.0
        self.contamination_hint_ = 0.1

    def _contamination_signal(self, scores):
        """Estimate the anomaly fraction from an unlabeled score sample.

        Placeholder: fixed 10% guess, ignoring `scores` entirely. This is the
        hook the variant asks you to make real.
        """
        return 0.1

    def _raw_scores(self, X):
        s = np.zeros(X.shape[0], dtype=np.float64)
        for nn in self.views_:
            d, _ = nn.kneighbors(X)
            s += d.mean(axis=1)
        return s / max(1, len(self.views_))

    def fit(self, X):
        """Fit the multi-view reference sets on unlabeled training data."""
        from sklearn.neighbors import NearestNeighbors

        X = np.asarray(X, dtype=np.float64)
        rng = np.random.RandomState(SEED)
        n = X.shape[0]
        m = min(self.subsample, n)
        self.views_ = []
        for _ in range(self.n_views):
            idx = rng.choice(n, size=m, replace=False)
            nn = NearestNeighbors(n_neighbors=min(self.k, m))
            nn.fit(X[idx])
            self.views_.append(nn)
        probe = X[rng.choice(n, size=min(n, 2000), replace=False)]
        train_scores = self._raw_scores(probe)
        self.scale_ = float(np.median(train_scores)) + 1e-12
        self.contamination_hint_ = self._contamination_signal(train_scores)
        return self

    def decision_function(self, X):
        """Anomaly scores for X (higher = more anomalous)."""
        X = np.asarray(X, dtype=np.float64)
        return self._raw_scores(X) / self.scale_
