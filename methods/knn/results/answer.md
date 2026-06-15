# Weighted K-Nearest-Neighbor Imputation (KNNimpute), distilled

KNN imputation fills each missing entry of a table by borrowing from the rows most similar to
the one the hole sits in. For a row with a hole at coordinate `ℓ`, it finds the `K` training
rows closest to it — measured on the coordinates they both observe — and writes into the hole
the inverse-distance-weighted average of those neighbors' values at `ℓ`. It is the *local* end
of the imputation spectrum: unlike per-column mean imputation it uses the observed part of each
row, and unlike global low-rank methods it fits no global model, just a local average that
follows the data's structure wherever the row lives.

## Problem it solves

Complete a tabular dataset `X` of shape `(n_samples, n_features)` whose entries may be `NaN`,
producing finite values for missing entries in features that have training observations —
learnable from observed entries only, applied by the *same* frozen rule out of sample, and
exploiting inter-feature dependence so that the fill for a hole depends on *which row* it is
in, not only which column.

## Key idea

For each missing entry, take a weighted average over the `K` nearest rows, where similarity is
a Euclidean distance restricted to co-observed coordinates and rescaled to full dimension:

- **Partial (nan-Euclidean) distance.** Euclidean distance is undefined when a coordinate is
  missing, so compute the squared distance over the coordinates present in *both* rows and
  scale it up by the ratio of total to present coordinates (Dixon, 1979):

  ```
  d(x, y)^2 = (D / |O|) * sum_{l in O} (x_l - y_l)^2 ,   O = { l : x_l, y_l both observed }
  ```

  with `D = n_features` and `|O| = |O(x,y)|` the overlap count. The `D/|O|` factor puts pairs
  with different amounts of overlap on a common, full-dimensional footing (without it, more
  overlapping pairs accumulate more squared terms and look farther). It reduces to ordinary
  Euclidean when nothing is missing, and is undefined (`NaN`) when `|O| = 0`.

- **Euclidean over correlation, because the data is log-scaled.** Among candidate similarity
  measures (Pearson correlation, Euclidean, variance-minimization), Euclidean is chosen.
  Squaring makes Euclidean outlier- and scale-sensitive, which would argue for level-invariant
  correlation — but log-transforming the (heavy-tailed, positive) measurements compresses
  outliers and equalizes scales, defusing that fragility, while Euclidean keeps the coordinate
  *magnitudes* that a value-copying imputer needs (a neighbor at the right *level*, not merely
  the right *shape*).

- **Inverse-distance weighting.** Among the `K` neighbors, weight each by `1/d`: the nearest is
  a strong witness, the `K`-th (the farthest selected) a weak one. The fill is

  ```
  x_hat_l = ( sum_{y in NN_K} (1/d(x,y)) * y_l ) / ( sum_{y in NN_K} 1/d(x,y) )
  ```

  If one or more selected donors have `d = 0`, only those exact-distance donors receive
  positive weight; multiple exact matches are averaged. This also makes the estimate
  **insensitive to `K`**: boundary neighbors are the farthest and carry the least weight, so
  adding/removing them barely moves the average — a broad sweet spot rather than a knife-edge.

- **`K`.** A moderate value works; the method is insensitive to it over a wide band (small `K`
  is noisy; large `K` drifts toward a global average and loses locality). Microarray use often
  chose values around `10`–`20`; the sklearn configuration here uses `n_neighbors = 5`.

- **Column-mean fallback.** For a missing coordinate `ℓ`, the donor pool is restricted to
  training rows that observe `ℓ`. If none of those donors has a defined distance to the
  receiver because every overlap is empty (`|O| = 0`), fill with the training column mean. The
  method is KNN where there is local evidence and the mean baseline where there is not.

- **Train/test discipline.** Neighbors are searched in the *training* rows and the fallback
  means are the *training* means, both frozen at `fit`; `transform` replays them on any table.
  `fit` never touches the target.

## Final algorithm

```
fit(X_train):
    store X_train, its missing-mask, valid feature mask, and per-column observed means

transform(X):
    for each row x with any hole:
        for each missing coordinate l of x among fit-time valid features:
            donors = training rows that observe l
            d = scaled nan-Euclidean distance from x to each donor   # (D/|O|) * overlap sum, sqrt
            keep donors with defined distance; if none -> x_hat_l = column_mean[l]; continue
            NN_K = K donors with smallest d
            w = 1/d over NN_K  (exact matches: weight 1, others 0)
            x_hat_l = weighted average of donors' l-values with weights w
    return completed X over fit-time valid features
```

## Working code

The scikit-learn form configures the library KNN imputer, which packages the nan-Euclidean
partial distance, `K`-nearest selection, inverse-distance weights, and column-mean fallback:

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer


class CustomImputer(BaseEstimator, TransformerMixin):
    """Weighted K-nearest-neighbor imputation."""

    def __init__(self, random_state=42, max_iter=10):
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_neighbors = 5

    def fit(self, X, y=None):
        self._imputer = KNNImputer(
            n_neighbors=self.n_neighbors,
            weights="distance",        # weight neighbor y by 1/d(x, y)
            metric="nan_euclidean",    # (D/|O|) * sum over co-observed coords, then sqrt
        )
        self._imputer.fit(X)           # stores train rows + masks + column means
        return self

    def transform(self, X):
        return self._imputer.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
```

The explicit form makes every step visible and matches the library behavior:

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances


class CustomImputerExplicit(BaseEstimator, TransformerMixin):
    """Scaled nan-Euclidean partial distance, K nearest donors per missing
    coordinate, inverse-distance weighted average, column-mean fallback."""

    def __init__(self, n_neighbors=5, keep_empty_features=False):
        self.n_neighbors = n_neighbors
        self.keep_empty_features = keep_empty_features

    def fit(self, X, y=None):                      # y unused
        X = np.asarray(X, dtype=float)
        self._fit_X = X                            # frozen training donor pool
        self._mask_fit = np.isnan(X)
        self._valid_mask = ~np.all(self._mask_fit, axis=0)
        counts = (~self._mask_fit).sum(axis=0)
        sums = np.nansum(X, axis=0)
        self._col_mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        return self

    def _nan_euclidean(self, A):
        # dist(x, y)^2 = (D / |O|) * sum_{l in O} (x_l - y_l)^2
        X, Y = np.asarray(A, dtype=float).copy(), np.asarray(self._fit_X, dtype=float).copy()
        missing_X, missing_Y = np.isnan(X), self._mask_fit
        X[missing_X] = 0.0
        Y[missing_Y] = 0.0
        distances = euclidean_distances(X, Y, squared=True)  # zero-filled squared distance
        XX, YY = X * X, Y * Y
        distances -= np.dot(XX, missing_Y.T)       # drop x_l^2 where y missing at l
        distances -= np.dot(missing_X, YY.T)       # drop y_l^2 where x missing at l
        np.clip(distances, 0.0, None, out=distances)
        present = np.dot(1 - missing_X, (~missing_Y).T)      # |O|
        distances[present == 0] = np.nan           # empty overlap -> undefined
        np.maximum(1, present, out=present)
        distances /= present
        distances *= X.shape[1]
        np.sqrt(distances, out=distances)
        return distances

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Xc = X.copy()
        mask = np.isnan(X)
        valid = self._valid_mask
        rows = np.where(mask[:, valid].any(axis=1))[0]
        if rows.size:
            dist = self._nan_euclidean(X[rows])     # (holey rows, n_train)
            k = self.n_neighbors
            for r, i in enumerate(rows):
                for col in np.where(mask[i] & valid)[0]:
                    donors = np.where(~self._mask_fit[:, col])[0]   # donors observing this column
                    d = dist[r, donors]
                    ok = ~np.isnan(d)
                    if not ok.any():
                        Xc[i, col] = self._col_mean[col]            # mean fallback
                        continue
                    donors, d = donors[ok], d[ok]
                    kk = min(k, len(donors))
                    sel = np.argpartition(d, kk - 1)[:kk]           # K nearest donors
                    dk, vk = d[sel], self._fit_X[donors[sel], col]
                    w = (dk == 0).astype(float) if np.any(dk == 0) else 1.0 / dk
                    Xc[i, col] = np.average(vk, weights=w)          # inverse-distance weighted mean
        if self.keep_empty_features:
            Xc[:, ~valid] = 0.0
            return Xc
        return Xc[:, valid]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
```

## Relation to prior methods

- **Column-mean imputation** = the no-covariate fallback, and also the uniform-weight,
  all-donor limit for a feature; the local rule reduces to it where no donor distance is
  defined.
- **Global low-rank (eigen-vector) imputation** = the *global* end of the spectrum (fit a few
  basis directions to the bulk and project); KNN is the *local* end (no global model, just
  nearby rows), so it handles unusual rows that the leading directions misfit.
- **Iterative regression / EM imputation** models each feature on the others and iterates; KNN
  is a single non-parametric pass with no per-feature model and no iteration.
- **Partial distance** is Dixon's (1979) strategy for distances under missingness; the
  nearest-neighbor rule itself goes back to Cover & Hart (1967).
