A table of numbers arrives with holes in it: some cells are `NaN`, and almost nothing downstream tolerates that. A regression routine silently drops any row carrying a hole, a gradient learner cannot even hold a `NaN` in a tensor, and a distance or kernel between two rows is undefined the moment one coordinate is absent. So the job is to turn an $(n, d)$ table-with-holes into an $(n, d)$ table of finite numbers, learning the rule only from the entries that are actually present, never peeking at the prediction target, and freezing that rule so a new row arriving later with its own holes is completed by exactly the same map — otherwise it is not a well-defined predictor. The cheapest honest answer is mean imputation: write into each hole of a column its observed mean. That is the least-squares constant guess for the column and the unique constant that leaves the column mean intact, so among no-covariate answers it is optimal. But "no covariate" is precisely the phrase that should bother us. To fill a hole in column three, mean imputation looks only at the other observed values in column three and ignores everything else about the row the hole sits in; two rows that differ wildly in every other coordinate receive the *same* number written into their column-three hole. That throws away the one thing that makes a hole guessable. Features in a real table are correlated — if column three tends to move with columns one and two, a row with large columns one and two probably has a large column three — and the column mean, blind to those coordinates, splits the difference and is wrong for both. The global rivals fall short from the other direction: a low-rank eigenvector completion fits a few basis directions to the *bulk* of the data and misfits any unusual row whose structure the leading directions do not capture, and it must guess a rank $J$; iterative chained-equation regression fits a parametric model per incomplete feature and grinds through slow EM iterations, placing every imputed point flat on a regression surface. We want a completion that reads the observed part of each row and uses it, without committing to any global or per-feature parametric model.

I propose weighted K-nearest-neighbor imputation. The most assumption-light way to predict a value at a point is to look at the *nearby* points and copy what they do — the nearest-neighbor rule of Cover and Hart (1967), whose single-nearest-neighbor classifier has asymptotic error at most twice the Bayes error, a remarkable thing to get for free that says the closest point already carries most of the answer. One neighbor is jumpy, so averaging the $K$ closest damps the variance and turns the prediction into a *local average* of the response surface, bending to follow whatever the surface does in this region with no global functional form. Concretely, to fill the hole at coordinate $\ell$ of a target row $x$, we find the $K$ training rows most similar to $x$ and average their values at coordinate $\ell$. "Most similar" demands a distance, and here the missingness bites one level up: the obvious squared Euclidean distance $\sum_\ell (x_\ell - y_\ell)^2$ runs over every coordinate, but $x$ is missing some — that is why we are here — and candidate rows may be missing others, so the term $(x_\ell - y_\ell)^2$ is undefined wherever either side has a hole. We cannot form the distance we need to find the neighbors that would fill the hole. The escape is to ask which part of the distance we *can* compute. For a pair $(x, y)$ some coordinates are observed in both — call that overlap set $O$ — and on those the squared difference is well-defined, so we could simply sum over the overlap. But that raw sum is not comparable across pairs: each squared term is nonnegative, so a pair sharing twenty observed coordinates tends to a larger total than a pair sharing five even if the latter is genuinely more similar per coordinate, and the comparison gets driven by how many coordinates the overlap happens to contain rather than by how close the rows actually are. What we want is an estimate of the *full* $d$-dimensional squared distance had nothing been missing. If the unseen coordinates contribute on average like the seen ones — the only thing we can assume without more information — then the per-coordinate average squared difference is $\frac{1}{|O|}\sum_{\ell \in O}(x_\ell - y_\ell)^2$, and the full sum over all $d$ coordinates is that average times $d$. This is the partial (nan-Euclidean) distance,
$$d(x, y)^2 = \frac{d}{|O|}\sum_{\ell \in O}(x_\ell - y_\ell)^2, \qquad O = \{\ell : x_\ell, y_\ell \text{ both observed}\}.$$
The $d/|O|$ scale-up is exactly the correction the raw overlap sum was missing: it puts the five-overlap and twenty-overlap pairs on a common full-dimensional footing. Check the boundaries — a fully observed row has $|O| = d$, the factor is $1$, and we recover ordinary squared Euclidean, which is the correct behavior when nothing is missing; and $|O| = 0$ gives $0/0$, undefined, correctly refusing to call two rows neighbors when they share no observed coordinate at all.

Is Euclidean even the right notion of similarity? Its serious weakness is that squaring each difference lets a single high-spread coordinate or one wild outlier dominate the entire sum and drown out every other coordinate, which argues for Pearson correlation between the two rows' observed profiles — invariant to a row's overall level and scale, capturing shape rather than absolute values — or a variance-minimization criterion that favors donors agreeing tightly on the imputed coordinate. The resolution is that the data is log-transformed before any analysis, as heavy-tailed positive measurements spanning orders of magnitude routinely are. The log compresses large values and pulls in the long right tail (a factor of a thousand becomes a factor of $\log 1000 \approx 7$) and turns multiplicative spread into additive spread, bringing wildly different scales onto a comparable footing — so the very outlier-and-scale fragility that made us distrust Euclidean is largely defused *before* the distance is ever computed, by a transform the data wants for independent reasons. On log-scaled data Euclidean is no longer hostage to a few extremes, and it keeps what correlation throws away: the actual coordinate *magnitudes*. Two rows can be perfectly correlated in shape yet sit at very different levels, and to copy a value into a hole we want a neighbor at the right *level*, not merely the right shape. Variance-minimization is worse still because it lets the donor coordinate we are estimating influence who counts as a donor, whereas a row-to-row distance uses only the observed profile. So once robustness is handled upstream, Euclidean wins.

Among the $K$ nearest donors, a flat mean would let the $K$-th neighbor — the farthest of the selected set, a weak witness that barely scraped in — pull as hard as the near-twin at the front. We therefore weight each neighbor by the inverse of its distance, forming the fill
$$\hat{x}_\ell = \frac{\sum_{y \in NN_K}(1/d(x, y))\, y_\ell}{\sum_{y \in NN_K} 1/d(x, y)},$$
so that "twice as far, half the say": the near-twin dominates and the marginal far neighbor barely registers. This weighting pays off a second time on the choice of $K$. With a flat average $K$ is a knife-edge — a neighbor enters at full weight or not at all — but with inverse-distance weights the boundary neighbors are the *farthest* and carry the *smallest* weights, so adding the $(K{+}1)$-th donor (necessarily farther than all $K$ already in) appends a term with weight smaller than every weight present and barely moves the estimate. The result is insensitive to the exact $K$ over a broad band: too small is noisy and swung by one odd neighbor, too large drifts toward a global weighted average and loses locality, but a moderate value in the low tens sits comfortably in the sweet spot for tables of a few hundred to a few thousand rows, and a small default like $5$ is fine. The boundary cases the partial distance warned about are handled honestly. For a missing coordinate $\ell$ the donor pool is first restricted to training rows that actually observe $\ell$, since a row missing $\ell$ has nothing to donate. The common failure is that donors exist but every one has empty overlap with the observed coordinates of $x$, so every partial distance is $0/0$ and there is no basis for ranking; then we fall back to the thing we started from — when we cannot be local, be global — and fill with the training column mean, the best no-covariate guess and exactly what we would have done with no usable neighbors. The method is thus KNN where there is local evidence and the mean baseline where there is not. When some selected donors have distance exactly zero, only those exact matches receive positive weight, so duplicate observed profiles decide the fill directly. All of this respects out-of-sample discipline: `fit` stores the training rows, their missingness mask, and the per-column observed means, all frozen; `transform` computes partial distances from any row *into the stored training rows*, never finding neighbors among test rows or recomputing means on the test set, and never touching the target $y$. Finally, the partial distance is computed vectorized rather than pair-by-pair: temporarily zero-fill the missing entries and take the ordinary squared Euclidean distance, then subtract back the spurious terms $\sum_\ell x_\ell^2[y \text{ missing at } \ell]$ and $\sum_\ell y_\ell^2[x \text{ missing at } \ell]$ that the substituted zeros introduced, leaving the genuine overlap sum as a couple of matrix products; divide by the overlap count (clamped to at least one, with zero-overlap pairs flagged as undefined) and multiply by $d$, then take the square root. That makes the whole thing a few dense linear-algebra operations and tractable at scale.

In scikit-learn this exact construction is already packaged as a transformer — the nan-aware Euclidean distance with the $d/|O|$ scale-up, the $K$-nearest selection, the inverse-distance weighting, and the column-mean fallback — so the faithful implementation configures it with `n_neighbors = 5` and distance weighting, learned on the training rows and replayed on any table:

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

The library wrapper hides the moving parts, so the explicit form writes each one out — the scale-up, the weighting, the fallback — and matches the same numerical behavior:

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
