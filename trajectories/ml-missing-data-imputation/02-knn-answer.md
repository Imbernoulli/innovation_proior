**Problem (from step 1).** The mean fill is blind to the row: it writes one number per column, so a hole gets the column average regardless of the row's other observed coordinates, discarding the inter-feature correlation that makes a hole guessable. RMSE suffered most on the wide correlated matrices (Breast Cancer 0.994, Wine 1.033, both above the unit-variance scale); downstream barely moved.

**Key idea.** Read the row. To fill the hole at coordinate `ℓ` of a target row, find the `K` training rows most similar on the *co-observed* coordinates and fill with their values at `ℓ`. A local average follows the response surface with no global model. Similarity is a distance, but Euclidean is undefined with holes, so use the scaled nan-Euclidean partial distance over the overlap set `O`, `d² ≈ (D/|O|)·Σ_{ℓ∈O}(x*_ℓ − y_ℓ)²` — comparable across pairs with different amounts of overlap, equal to ordinary Euclidean when nothing is missing. The data is pre-standardized to unit variance, so Euclidean is not scale-dominated and keeps the magnitudes a copied fill needs.

**Why.** Inverse-distance weighting (`weights="distance"`) lets a near-twin dominate and a marginal far neighbour barely register, which also makes the result insensitive to the exact `K`. Where no donor has a defined distance, `KNNImputer` falls back to the training column mean — so the method degrades to the step-1 floor where local evidence is absent and only improves where it is present. Fit-time neighbours and means are frozen and replayed; labels never used.

**Scaffold edit / hyperparameters.** Fill the `CustomImputer` slot with `sklearn.impute.KNNImputer(n_neighbors=5, weights="distance")` (default `metric="nan_euclidean"`). `n_neighbors=5`: low single digits is local but averaged, and the distance weighting makes the band forgiving.

**What to watch.** RMSE should fall most on Breast Cancer / Wine. The risk is California: with only 8 features the overlap set is tiny, the `D/|O|` scale-up extrapolates from little, and a local average over noisy neighbours of a continuous regression target can be *worse* than the stable column mean — RMSE may regress above 0.928 and R² below 0.646.

```python
class CustomImputer(BaseEstimator, TransformerMixin):
    """KNN Imputation: impute using K-nearest neighbors.

    Uses sklearn.impute.KNNImputer with n_neighbors=5, distance weighting.
    Reference: Troyanskaya et al. (2001).
    """

    def __init__(self, random_state=42, max_iter=10):
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_neighbors = 5

    def fit(self, X, y=None):
        from sklearn.impute import KNNImputer
        self._imputer = KNNImputer(
            n_neighbors=self.n_neighbors,
            weights="distance",
        )
        self._imputer.fit(X)
        return self

    def transform(self, X):
        return self._imputer.transform(X)

    def fit_transform(self, X, y=None):
        from sklearn.impute import KNNImputer
        self._imputer = KNNImputer(
            n_neighbors=self.n_neighbors,
            weights="distance",
        )
        return self._imputer.fit_transform(X)


def compute_feature_correlations(X):
    n_features = X.shape[1]
    corr = np.eye(n_features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
            if mask.sum() > 2:
                c = np.corrcoef(X[mask, i], X[mask, j])[0, 1]
                corr[i, j] = corr[j, i] = c if not np.isnan(c) else 0.0
    return corr
```
