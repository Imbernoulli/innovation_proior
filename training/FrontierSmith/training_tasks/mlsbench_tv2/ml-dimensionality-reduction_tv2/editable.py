class CustomDimReduction:
    """Class-structure-first 2-D embedding via manufactured pseudo-labels.

    This variant is scored the same three ways as always but designed for
    one of them: the 7-NN accuracy a classifier reaches in the output map.
    No labels are visible at fit time, so the scaffold builds a surrogate
    partition and lays the map out to separate it:

      1. PCA sketch to `n_sketch` dims (denoise + speed).
      2. K-Means over the sketch -> `n_pseudo` pseudo-classes.
      3. A linear discriminant projection to 2-D fitted on pseudo-labels.

    The pseudo-partition is the lever. Sharper unsupervised group structure
    (better surrogates, mutual-kNN filtering, density modes, ensembles of
    partitions) should translate directly into 7-NN accuracy; the two
    fidelity scores act as guard rails against degenerate layouts.

    Interface (fixed): __init__(n_components=2, random_state=None);
    fit_transform(X) -> (n_samples, n_components), finite values.
    numpy/scipy/scikit-learn may be imported inside methods.
    """

    n_sketch = 40      # PCA sketch width feeding the pseudo-labeler
    n_pseudo = 20      # surrogate classes to manufacture

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state
        self.pseudo_labels_ = None

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Embed X to n_components dims, optimizing for downstream 7-NN."""
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        m = int(min(self.n_sketch, d, max(2, n - 1)))
        Z = PCA(n_components=m, random_state=self.random_state).fit_transform(X)

        k = int(min(self.n_pseudo, max(3, n // 100)))
        lab = KMeans(n_clusters=k, random_state=self.random_state,
                     n_init=4).fit_predict(Z)
        self.pseudo_labels_ = lab

        try:
            Y = LinearDiscriminantAnalysis(
                n_components=self.n_components).fit_transform(Z, lab)
        except Exception:
            Y = Z[:, : self.n_components]
        if Y.shape[1] < self.n_components:   # degenerate surrogate partition
            Y = np.hstack([Y, np.zeros((n, self.n_components - Y.shape[1]))])
        return np.ascontiguousarray(Y, dtype=np.float64)
