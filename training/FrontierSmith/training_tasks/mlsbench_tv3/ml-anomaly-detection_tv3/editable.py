class CustomAnomalyDetector:
    """Alert-budget scoring: protect the top slice of the ranking.

    Objective of this variant: F1 at the contamination threshold is decided
    by the few test points ranked above the operating line — the alert
    budget. This baseline refuses to spend an alert on one view's say-so:
    a point ranks high only when an isolation view and a distance view
    AGREE, which keeps single-view artefacts (one spectacular coordinate,
    one idiosyncratic heavy tail) out of the top slice.

    Interface (fixed):
        __init__(self)               -- no required arguments
        fit(self, X) -> self         -- unsupervised, X standardized
        decision_function(self, X)   -- (n_samples,) scores, higher = anomalous

    Structure provided as the intended adaptation channels:
      - two deliberately dissimilar component views: a pyod IForest and the
        k-th-neighbour distance to a fitted reference subsample.
      - `_to_rank(sorted_ref, s)`: train-referenced rank in [0, 1] via
        stored sorted train scores — the common currency both views are
        converted into before fusion.
      - `_fuse(r_iso, r_dist)`: placeholder consensus = elementwise minimum
        of the two rank profiles. Budget-aware fusion (weighted consensus,
        top-slice de-duplication, threshold-location estimation) is the
        intended upgrade.
      - `budget_`: a fixed 0.05 guess at the alert fraction; recorded but
        never used — estimating it from unlabeled scores is part of the
        task.

    Available libraries: numpy, scipy, scikit-learn, pyod (import inside
    methods).
    """

    def __init__(self, ref_size=2048, k=10, probe_size=4096):
        self.ref_size = ref_size
        self.k = k
        self.probe_size = probe_size
        self.budget_ = 0.05
        self.iforest_ = None
        self.nn_ = None
        self.ref_iso_ = None
        self.ref_dist_ = None

    def _kth_dist(self, X):
        d, _ = self.nn_.kneighbors(X)
        return d[:, -1]

    @staticmethod
    def _to_rank(sorted_ref, s):
        """Rank of each score against the stored train scores, in [0, 1]."""
        pos = np.searchsorted(sorted_ref, s, side="left")
        return pos / max(1, len(sorted_ref))

    @staticmethod
    def _fuse(r_iso, r_dist):
        """Consensus: a point is only as anomalous as its weakest view."""
        return np.minimum(r_iso, r_dist)

    def fit(self, X):
        """Fit both views and store their train score distributions."""
        from pyod.models.iforest import IForest
        from sklearn.neighbors import NearestNeighbors

        X = np.asarray(X, dtype=np.float64)
        rng = np.random.RandomState(SEED)
        n = X.shape[0]
        self.iforest_ = IForest(random_state=SEED)
        self.iforest_.fit(X)
        ref = X[rng.choice(n, size=min(self.ref_size, n), replace=False)]
        self.nn_ = NearestNeighbors(n_neighbors=min(self.k, len(ref))).fit(ref)
        probe = X[rng.choice(n, size=min(self.probe_size, n), replace=False)]
        self.ref_iso_ = np.sort(self.iforest_.decision_function(probe))
        self.ref_dist_ = np.sort(self._kth_dist(probe))
        return self

    def decision_function(self, X):
        """Consensus rank score (high only if both views call it anomalous)."""
        X = np.asarray(X, dtype=np.float64)
        r_iso = self._to_rank(self.ref_iso_, self.iforest_.decision_function(X))
        r_dist = self._to_rank(self.ref_dist_, self._kth_dist(X))
        return self._fuse(r_iso, r_dist)
