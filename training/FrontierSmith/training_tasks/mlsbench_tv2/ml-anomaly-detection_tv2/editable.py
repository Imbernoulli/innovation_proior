class CustomAnomalyDetector:
    """Anomaly scoring from a recovered clean core of a polluted fit set.

    Objective of this variant: the training split is untrusted — it carries
    the dataset's own anomaly fraction (up to ~31.6% on satellite), so any
    model of "normal" fitted on it verbatim is partly a model of the
    anomalies. This detector therefore separates a provisional clean core
    from the fit sample before scoring, and scores test points by distance
    to that core only.

    Interface (fixed):
        __init__(self)               -- no required arguments
        fit(self, X) -> self         -- unsupervised, X standardized
        decision_function(self, X)   -- (n_samples,) scores, higher = anomalous

    Structure provided as the intended adaptation channels:
      - `trim`: fraction of the reference sample discarded as provisional
        anomalies after a self-scoring pass. Fixed at 0.10 here — a single
        hard trim at a guessed rate. Estimating how much to trim per
        dataset, trimming softly (weights instead of deletion), or
        iterating score -> trim -> refit to a fixed point are the intended
        upgrades.
      - `passes`: number of trim/refit rounds (placeholder: 1).
      - `core_` / `scale_`: the retained reference points and the median
        core self-score used to normalise test scores.

    Available libraries: numpy, scipy, scikit-learn, pyod (import inside
    methods).
    """

    def __init__(self, ref_size=2048, k=8, trim=0.10, passes=1):
        self.ref_size = ref_size
        self.k = k
        self.trim = trim
        self.passes = passes
        self.core_ = None
        self.scale_ = 1.0

    def _mean_knn_dist(self, ref, X, exclude_self=False):
        """Mean distance from rows of X to their k nearest points in ref."""
        from sklearn.neighbors import NearestNeighbors

        kk = min(self.k + (1 if exclude_self else 0), len(ref))
        nn = NearestNeighbors(n_neighbors=kk).fit(ref)
        d, _ = nn.kneighbors(X)
        if exclude_self:
            d = d[:, 1:]
        return d.mean(axis=1)

    def fit(self, X):
        """Recover a clean core from the polluted, unlabeled fit sample."""
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.RandomState(SEED)
        n = X.shape[0]
        ref = X[rng.choice(n, size=min(self.ref_size, n), replace=False)]
        for _ in range(max(1, int(self.passes))):
            self_scores = self._mean_knn_dist(ref, ref, exclude_self=True)
            keep = self_scores <= np.quantile(self_scores, 1.0 - self.trim)
            if keep.sum() >= self.k + 1:
                ref = ref[keep]
        self.core_ = ref
        core_scores = self._mean_knn_dist(ref, ref, exclude_self=True)
        self.scale_ = float(np.median(core_scores)) + 1e-12
        return self

    def decision_function(self, X):
        """Score = normalised mean distance to the recovered clean core."""
        X = np.asarray(X, dtype=np.float64)
        return self._mean_knn_dist(self.core_, X) / self.scale_
