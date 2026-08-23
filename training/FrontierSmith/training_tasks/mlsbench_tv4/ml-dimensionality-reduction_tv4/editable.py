class CustomDimReduction:
    """Compute-frugal embedding: landmarks + power iteration, nothing heavy.

    The discipline of this variant is arithmetic rationing. The scaffold
    holds itself to a self-imposed wall-clock allowance (`budget_s`) far
    below the pipeline limit, and to simple auditable algebra:

      * `m_landmarks` rows sampled once stand in for the full input;
      * the top directions of the landmark covariance are found by plain
        power iteration with deflation -- matrix-vector products only, no
        decompositions;
      * every row is projected onto those directions.

    Memory never holds an n x n object and cost grows ~linearly with n.
    Better frugal machinery -- Nystrom extensions, sparse neighbor stubs on
    landmarks only, few-step fixed-point refinements -- should buy metric
    quality per second, and every addition must respect the budget guard.

    Interface (fixed): fit_transform(X) -> (n_samples, n_components),
    finite values, reproducible for a given random_state.
    """

    budget_s = 60.0      # self-imposed wall-clock allowance (seconds)
    m_landmarks = 512    # rows that stand in for the full dataset
    n_power = 60         # max power-iteration steps per direction

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Embed X via landmark covariance directions, within budget."""
        import time

        t0 = time.time()
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape

        m = int(min(self.m_landmarks, n))
        idx = rng.choice(n, size=m, replace=False)
        L = X[idx]
        mu = L.mean(axis=0)
        Lc = L - mu
        C = (Lc.T @ Lc) / max(1, m - 1)           # d x d from landmarks only

        dirs = []
        for _ in range(self.n_components):
            v = rng.standard_normal(d)
            v /= np.linalg.norm(v) + 1e-12
            for _ in range(self.n_power):
                if time.time() - t0 > self.budget_s:
                    break
                for u in dirs:                    # deflate found directions
                    v = v - (v @ u) * u
                w = C @ v
                nrm = np.linalg.norm(w)
                if nrm < 1e-12:
                    break
                v = w / nrm
            for u in dirs:
                v = v - (v @ u) * u
            v /= np.linalg.norm(v) + 1e-12
            dirs.append(v)

        W = np.stack(dirs, axis=1)                # d x n_components
        return np.ascontiguousarray((X - mu) @ W, dtype=np.float64)
