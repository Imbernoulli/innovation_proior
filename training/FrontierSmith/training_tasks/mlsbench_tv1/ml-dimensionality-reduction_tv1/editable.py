class CustomDimReduction:
    """Symmetric-fidelity 2-D embedding under one configuration and a budget.

    Objective of this variant: keep BOTH local error directions low at once —
    few invented neighbors (trustworthiness) and few torn neighborhoods
    (continuity) — with a single fixed configuration that must serve raw
    784-D pixel data and 50-D SVD-compressed text alike, finishing within the
    pipeline's per-dataset CPU budget. Adaptation must come from statistics
    computed on the given data (see `_geometry_probe`), not per-dataset
    switches.

    Interface (fixed):
        __init__(n_components=2, random_state=None)
        fit_transform(X) -> (n_samples, n_components), finite values,
        reproducible for a given random_state.

    Structure provided as the intended adaptation channels:
      - `_geometry_probe(X, rng)`: a scale/geometry statistic of the input
        (placeholder: median k-NN distance of a subsample). It is stored in
        self.probe_ and currently only rescales the output — wiring it into
        neighborhood size, step size, or attraction/repulsion balance is the
        intended use.
      - `time_budget_s` + the iteration guard: a hard wall-clock budget for
        the refinement loop; any replacement optimizer must keep the guard.
      - the refinement stub: `n_iter` sweeps of neighborhood smoothing that
        pull each point toward the embedding centroid of its ORIGINAL-space
        neighbors (continuity-oriented). It has no repulsion, so it slowly
        contracts clusters — trustworthiness is under-served on purpose.

    numpy/scipy/scikit-learn are available (import inside the method).
    """

    # Hard wall-clock budget (seconds) for one fit_transform call.
    time_budget_s = 240.0
    # Neighborhood size used by the probe and the smoothing stub.
    n_neighbors = 10
    # Placeholder refinement sweeps (deliberately few, attraction-only).
    n_iter = 6
    # Smoothing step toward the neighbor centroid, in (0, 1).
    step = 0.25

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state
        self.probe_ = None

    def _geometry_probe(self, X, rng):
        """Median distance to the k-th nearest neighbor over a subsample.

        A continuous, label-free description of the input's local scale;
        small for tight pixel manifolds, larger for diffuse SVD text vectors.
        Placeholder use is cosmetic (global rescale) — this is the intended
        adaptation channel.
        """
        from sklearn.neighbors import NearestNeighbors

        n = X.shape[0]
        m = min(n, 512)
        idx = rng.choice(n, size=m, replace=False)
        k = min(self.n_neighbors + 1, m)
        d, _ = NearestNeighbors(n_neighbors=k).fit(X[idx]).kneighbors(X[idx])
        return float(np.median(d[:, -1])) + 1e-12

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Embed X into n_components dimensions."""
        import time
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors

        t0 = time.time()
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        self.probe_ = self._geometry_probe(X, rng)

        # Linear spectral initialization.
        Y = PCA(n_components=self.n_components,
                random_state=self.random_state).fit_transform(X)

        # Budget-guarded refinement stub: attraction-only smoothing toward the
        # embedding centroid of each point's original-space neighbors.
        k = min(self.n_neighbors + 1, X.shape[0])
        _, nbr = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
        nbr = nbr[:, 1:]
        for _ in range(self.n_iter):
            if time.time() - t0 > self.time_budget_s:
                break
            Y = (1.0 - self.step) * Y + self.step * Y[nbr].mean(axis=1)

        return Y / self.probe_
