# ================================================================
# EDITABLE -- agent modifies this section
# ================================================================
# Variant: geometry first. Silhouette leads the design; refinements are
# accepted only when they improve intrinsic quality, with a structural
# floor on the cluster count guarding against extrinsic collapse.


class CustomClustering(BaseEstimator, ClusterMixin):
    """Intrinsic-quality-first clusterer with an anti-collapse guard.

    Required interface (do not change signatures):
        fit(X) -> self          : sets self.labels_
        predict(X) -> labels    : (n_samples,) integer cluster ids

    Scaffold logic:
    - Start from a partitional fit at the hinted count.
    - Greedy geometric refinement: repeatedly merge the pair of clusters
      whose centers sit closest, keeping a merge only when the silhouette
      improves.
    - Anti-collapse guard: the count never drops below about half the
      starting count (see `k_floor` in fit), so compactness cannot be
      bought by dissolving true classes into super-clusters.
    - `self.sil_` records the achieved intrinsic score.

    Better levers than merging: metric shaping / embeddings that make true
    groups compact, margin-seeking reassignment of boundary points, and
    boundaries that follow density valleys -- each still subject to the
    guard.

    Args:
        n_clusters: starting count (a hint, refined geometrically).
        random_state: seed for reproducibility.
    """

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        self.sil_ = None

    @staticmethod
    def _silhouette(X, lab):
        try:
            if np.unique(lab).size < 2:
                return -np.inf
            return float(silhouette_score(X, lab))
        except ValueError:
            return -np.inf

    @staticmethod
    def _merge_closest(X, lab):
        """Merge the pair of clusters whose centers are closest."""
        ids = np.unique(lab)
        centers = np.stack([X[lab == c].mean(axis=0) for c in ids])
        d2 = ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        d2[np.diag_indices_from(d2)] = np.inf
        i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
        out = lab.copy()
        out[out == ids[j]] = ids[i]
        return out

    def fit(self, X):
        """Partition X, then refine the geometry under the collapse guard."""
        from sklearn.cluster import KMeans

        X = np.asarray(X, dtype=np.float64)
        k0 = self.n_clusters if self.n_clusters is not None else 8
        k0 = int(max(2, min(k0, X.shape[0] - 1)))
        lab = KMeans(n_clusters=k0, random_state=self.random_state,
                     n_init=10).fit_predict(X)
        best = self._silhouette(X, lab)

        k_floor = max(2, (k0 + 1) // 2)   # anti-collapse guard
        while np.unique(lab).size > k_floor:
            cand = self._merge_closest(X, lab)
            s = self._silhouette(X, cand)
            if s <= best:
                break
            lab, best = cand, s

        ids = np.unique(lab)              # compact ids to 0..C-1
        remap = np.zeros(int(ids.max()) + 1, dtype=np.int64)
        remap[ids] = np.arange(ids.size)
        self.labels_ = remap[lab]
        self.sil_ = best
        return self

    def predict(self, X):
        """Fitted labels for the training matrix (stateless refit otherwise)."""
        X = np.asarray(X, dtype=np.float64)
        if self.labels_ is not None and X.shape[0] == self.labels_.shape[0]:
            return self.labels_
        return self.fit(X).labels_


def custom_distance(x, y):
    """Distance hook -- shape THIS to make true groups geometrically tight."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    return float(np.sqrt(((a - b) ** 2).sum()))
