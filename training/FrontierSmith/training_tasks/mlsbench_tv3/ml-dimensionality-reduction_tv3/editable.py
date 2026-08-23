class CustomDimReduction:
    """Deterministic embedding: the map is a pure function of the data.

    Reproducibility is the design axis of this variant. `random_state` is
    accepted for interface compatibility, but the pipeline below never
    consumes randomness: identical inputs give bitwise-identical outputs.

    Scaffold (all closed-form, all deterministic):
      1. Center X.
      2. Eigendecompose the d x d covariance (np.linalg.eigh is
         deterministic for a given input).
      3. Resolve each top eigenvector's sign ambiguity by a canonical
         convention -- the coordinate of largest magnitude is made
         positive, first index winning exact ties.
      4. Project.

    This is only a linear floor. The intended work is a *nonlinear* stage
    that stays exactly repeatable -- deterministic neighbor graphs, fixed
    sweep orders, closed-form updates -- replacing the stochastic
    optimizers the reference methods rely on.

    Interface (fixed): fit_transform(X) -> (n_samples, n_components),
    finite values; scipy/scikit-learn importable inside methods if needed.
    """

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state  # deliberately unused downstream

    @staticmethod
    def _canonical_signs(V):
        """Fix eigenvector sign ambiguity with a data-independent rule."""
        W = V.copy()
        for j in range(W.shape[1]):
            i = int(np.argmax(np.abs(W[:, j])))   # first max wins ties
            if W[i, j] < 0:
                W[:, j] = -W[:, j]
        return W

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Deterministically embed X into n_components dimensions."""
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        Xc = X - X.mean(axis=0)

        C = (Xc.T @ Xc) / max(1, n - 1)           # d x d, d <= 784 here
        _, evecs = np.linalg.eigh(C)              # ascending eigenvalues
        top = evecs[:, ::-1][:, : self.n_components]
        top = self._canonical_signs(top)
        return np.ascontiguousarray(Xc @ top, dtype=np.float64)
