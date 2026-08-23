class CustomDimReduction:
    """Noise-aware embedding: score input dimensions, mute the rough ones.

    Premise of this variant: many input coordinates carry no structure
    (blank border pixels, tail SVD residue), and they dilute every distance
    the embedding relies on. The scaffold estimates per-feature relevance
    without labels and reweights the matrix before a linear embed:

      1. Subsample `n_sub` rows and build a small kNN graph (`k_graph`).
      2. Roughness of feature f = mean squared difference of f across graph
         edges; smooth features (low roughness) are structure-bearing.
      3. weight_f = 1 / (1 + damp * roughness_f / median roughness), so on
         uniformly clean inputs the weights approach uniform.
      4. PCA of the reweighted matrix -> 2-D.

    The weighting stage is the lever: better relevance estimators
    (Laplacian scores with proper degree normalization, spectral filtering,
    iterative reweighting with graph rebuilding) and a nonlinear stage on
    the cleaned geometry are the intended upgrades.

    Interface (fixed): fit_transform(X) -> (n_samples, n_components),
    finite values, reproducible under random_state.
    """

    n_sub = 600     # subsample used to score features
    k_graph = 8     # kNN graph size on the subsample
    damp = 4.0      # down-weighting strength for rough features

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state
        self.feature_weights_ = None

    def _feature_roughness(self, X, rng):
        """Per-feature mean squared edge difference over a subsample graph."""
        from sklearn.neighbors import NearestNeighbors

        n = X.shape[0]
        m = int(min(self.n_sub, n))
        S = X[rng.choice(n, size=m, replace=False)]
        k = int(min(self.k_graph + 1, m))
        _, nbr = NearestNeighbors(n_neighbors=k).fit(S).kneighbors(S)
        diff = S[nbr[:, 1:]] - S[:, None, :]      # (m, k-1, d)
        return (diff ** 2).mean(axis=(0, 1))

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reweight features by estimated relevance, then embed to 2-D."""
        from sklearn.decomposition import PCA

        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=np.float64)

        rough = self._feature_roughness(X, rng)
        med = float(np.median(rough)) + 1e-12
        w = 1.0 / (1.0 + self.damp * rough / med)
        self.feature_weights_ = w

        Y = PCA(n_components=self.n_components,
                random_state=self.random_state).fit_transform(X * w)
        return np.ascontiguousarray(Y, dtype=np.float64)
