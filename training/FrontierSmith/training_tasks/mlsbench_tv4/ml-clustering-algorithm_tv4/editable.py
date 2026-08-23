# ================================================================
# EDITABLE -- agent modifies this section
# ================================================================
# Variant: determinism discipline. The partition must be a function of the
# data; the seed is a formality. Consensus over restarts is the scaffold's
# stabilizer, and predict is a deterministic read-out of the fitted model.


class CustomClustering(BaseEstimator, ClusterMixin):
    """Seed-stable clusterer: same data in, same partition out.

    Required interface (do not change signatures):
        fit(X) -> self          : sets self.labels_
        predict(X) -> labels    : deterministic read-out, no refitting

    Stabilization scaffold:
    - `n_restarts` K-Means runs launch from distinct derived seeds; pairwise
      agreement (ARI between runs) selects the medoid partition -- the run
      every other run most agrees with. Restart noise is voted out rather
      than hidden behind one lucky seed.
    - `self.consensus_` stores the chosen run's mean agreement: a
      self-diagnosed stability report.
    - predict() assigns to the stored centers, so fit-then-predict on the
      same matrix reproduces self.labels_.

    Upgrades in spirit: co-association / evidence-accumulation consensus,
    deterministic seeding (farthest-first from a data-derived origin),
    agreement-filtered points, stability-weighted centers.

    Args:
        n_clusters: cluster-count metadata (used as given here).
        random_state: base seed; the design goal is to make it irrelevant.
    """

    n_restarts = 7

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        self.consensus_ = None
        self._centers_ = None

    def fit(self, X):
        """Vote a stable partition out of independent restarts."""
        from sklearn.cluster import KMeans

        X = np.asarray(X, dtype=np.float64)
        k = self.n_clusters if self.n_clusters is not None else 8
        k = int(max(2, min(k, X.shape[0] - 1)))
        base = 0 if self.random_state is None else int(self.random_state)

        runs = []
        for r in range(self.n_restarts):
            km = KMeans(n_clusters=k, random_state=base + 7919 * r, n_init=3)
            runs.append((km.fit_predict(X), km.cluster_centers_))

        n_runs = len(runs)
        agree = np.zeros(n_runs)
        for i in range(n_runs):
            for j in range(n_runs):
                if i != j:
                    agree[i] += adjusted_rand_score(runs[i][0], runs[j][0])
        best = int(np.argmax(agree))
        self.consensus_ = float(agree[best] / max(1, n_runs - 1))
        self.labels_ = runs[best][0]
        self._centers_ = runs[best][1]
        return self

    def predict(self, X):
        """Deterministic nearest-center read-out of the fitted model."""
        X = np.asarray(X, dtype=np.float64)
        d2 = ((X[:, None, :] - self._centers_[None, :, :]) ** 2).sum(axis=-1)
        return d2.argmin(axis=1)


def custom_distance(x, y):
    """Distance hook (kept deterministic; no hidden data-dependent state)."""
    diff = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(np.sqrt(diff @ diff))
