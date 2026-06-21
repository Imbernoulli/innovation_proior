The mean fill came in exactly where I predicted it would hurt: RMSE of $0.994$ on Breast Cancer and $1.033$ on Wine, both *above* the unit standard deviation of the z-scored columns — the tell that the constant fill is on average no closer to a masked cell than a random draw from the column — while the downstream scores held up almost embarrassingly well ($0.946$, $0.927$, $0.646$), because the `GradientBoosting` model re-fits on the completed matrix and launders each per-column spike into its own region. So the diagnosis is sharp and it is a *correlation* problem, not a scale or downstream-model problem: to fill a hole in column three the mean looks only at the other observed values *in column three*, writing the same number into every row's column-three hole no matter what that row is doing in its other coordinates. Two rows that differ wildly everywhere else get the identical fill, and that is the whole loss. The information about a missing entry is sitting in the same row, in the coordinates that row *does* observe — the thirty Breast Cancer columns are largely repeated geometric measurements of one tumour, so a row with large radius and perimeter almost certainly has large area, and the column mean for area is wrong for that row in a way its own observed coordinates could have corrected.

So the move is forced: stop filling from the column and start reading the row. I propose **K-nearest-neighbour imputation with a weighted nan-Euclidean distance**, in the lineage of Troyanskaya et al. (2001). To fill the hole at coordinate $\ell$ of a target row, find the $K$ training rows most similar to it and average *their* values at coordinate $\ell$. This is the oldest non-parametric idea there is — predict a value at a point by looking at nearby points and copying what they do — and I reach for it precisely because it commits to no parametric form for how the features relate, which I do not yet know to be linear. A single nearest neighbour is jumpy, so averaging over the $K$ closest damps that variance and turns the prediction into a local average of the response surface, bending to whatever structure exists in this region without my ever writing down a global function.

"Most similar" has to mean a distance, and reaching for ordinary squared Euclidean walks straight into the wall this task is about: $d(x^*, y)^2 = \sum_\ell (x^*_\ell - y_\ell)^2$ runs over every coordinate, but $x^*$ is missing some — that is why I am here — and the candidate rows may be missing some too, so the term $(x^*_\ell - y_\ell)^2$ is undefined wherever either side has a hole at $\ell$. The missingness has bitten me one level up. What I *can* compute, for a pair, is the part over the coordinates observed in both — the overlap set $O$ — and the naive move is to sum over $O$ and ignore the rest. But that raw sum is not comparable across pairs: if $x^*$ overlaps row $A$ on five coordinates and row $B$ on twenty, $B$'s sum tends to be larger simply because it has more nonnegative terms, which would rank the rows I have *more* evidence about as farther away — backwards. What I want is an estimate of the full $d$-dimensional distance had nothing been missing, so I assume the unseen coordinates contribute on average like the seen ones and scale the per-coordinate overlap average up to all $d$:
$$d(x^*, y)^2 \approx \frac{d}{|O|}\sum_{\ell \in O}(x^*_\ell - y_\ell)^2.$$
That $d/|O|$ factor is exactly the correction the raw sum lacked — it puts pairs sharing five and twenty coordinates on the same full-dimensional footing, reduces to ordinary squared Euclidean when nothing is missing ($|O| = d$, factor $1$), and goes correctly undefined when two rows share no observed coordinate ($|O| = 0$), where I have no basis to call them neighbours. Euclidean is the right notion of similarity here, more defensible than usual, because the substrate already z-scored every column to unit variance before punching holes: no single feature can dominate the sum by scale, which removes the main reason I would reach for a level-invariant metric like correlation. And Euclidean keeps the thing correlation throws away — the actual magnitudes — which is what I want, since I am about to copy a neighbour's actual value into the hole and need a neighbour at the right level, not merely the right shape.

The averaging is where the second design choice lives. Among the $K$ nearest, an equal vote lets the $K$-th neighbour — by construction the farthest of the chosen set, a weak witness that barely scraped in — pull as hard as the near-twin at the front. So I weight each neighbour by a decreasing function of its distance, the simplest being inverse distance ("twice as far, half the say"), and form the weighted average. This pays off twice: the near-twin dominates, *and* the estimate becomes insensitive to the exact $K$, because the boundary neighbours carry the least weight, so the answer is set by the handful of genuinely close rows. That means $K$ need not be tuned precisely — anywhere in a broad band the answer is stable; too small averages too few witnesses, too large drags in dissimilar rows and creeps back toward the global mean I am trying to beat. A small value in the low single digits sits comfortably in that band for matrices of a few hundred to a few thousand rows, so I take `n_neighbors=5` with distance weighting. The contract also demands a finite value for every hole, and the partial distance hands me the graceful degradation: for a missing coordinate the donor pool is the training rows that actually observe it, and if every candidate has empty overlap with the target's observed coordinates, every partial distance is undefined and the honest fallback is the training column mean — the best no-covariate guess. So the method degrades to exactly the step-1 floor where no local evidence exists and improves on it wherever local evidence is present; it cannot do worse on the cells where neighbours fail, only better where they succeed. The neighbours a row looks up and the means it falls back to are fixed at `fit` time and replayed unchanged, and the labels are never touched.

This exact construction — the partial distance with the $d/|O|$ scale-up, the $K$-nearest selection, the inverse-distance weighting, and the column-mean fallback — is precisely scikit-learn's `KNNImputer` with `metric="nan_euclidean"` (its default), so the faithful edit fills the `CustomImputer` slot with that transformer at `n_neighbors=5`, `weights="distance"`, fit on the training rows and replayed in `transform`. I expect the RMSE to fall most where the mean fill suffered most — the wide correlated Breast Cancer and Wine matrices, where reading the row pays — with the near-saturated downstream scores barely moving. The risk I can feel in the construction is California: with only eight features the overlap set per pair is tiny, typically six or seven coordinates, so the $d/|O|$ scale-up extrapolates a full-dimensional distance from very little, neighbour rankings get noisy, and a local average over noisy neighbours of a genuinely continuous, weakly-redundant regression target can be *worse* than the stable column mean — its RMSE may regress above $0.928$ and its R² below $0.646$. If California regresses, the next rung's diagnosis is already written: a purely local average over too few co-observed coordinates is the wrong tool when the table is narrow, and I will need a method that models each column from all the others jointly rather than copying nearby rows.

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
