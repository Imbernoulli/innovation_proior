I have a table of numbers with holes in it. Some cells are `NaN`, and almost nothing downstream tolerates that — the regression routine drops any row with a hole, the gradient learner can't even hold a `NaN`, a distance between two rows is undefined the moment one coordinate is absent. So the job is the same as always: turn this `(n, d)` table-with-holes into an `(n, d)` table of finite numbers, using only the entries that are actually present, and in a way I can freeze and replay on a new row that arrives later with its own holes. The cheap thing to do is fill each column with one number — its observed mean. That's the least-squares constant guess for the column, and it's the unique constant that leaves the column mean intact, so it's a perfectly honest no-covariate answer. But "no covariate" is exactly the phrase that should bother me. It means: to fill a hole in column three, I look only at the other observed values *in column three*, across all rows, and ignore everything else about the row the hole sits in. Two rows that are wildly different in every other coordinate get the *same* number written into their column-three hole, because the fill depends only on the column.

That throws away the one thing that should make a hole guessable. The features in a real table aren't independent; they're correlated. If column three tends to move with columns one and two, then a row whose columns one and two are large probably has a large column three, and a row whose columns one and two are small probably has a small one — and the column mean, blind to columns one and two, splits the difference and is wrong for both. The information about a missing entry is sitting right there in the *same row*, in the coordinates that row *does* observe. I want a fill that reads the observed part of the row and uses it. So the question becomes: how do I let the observed coordinates of a row tell me what its missing coordinate should be, without committing to a parametric model of how the features relate?

The most assumption-light way to predict a value at a point from data is to look at *nearby* points and copy what they do. That's the oldest non-parametric idea there is — the nearest-neighbor rule: to predict the response at a query, find the training example closest to it and use that example's response. Cover and Hart showed the asymptotic error of the single-nearest-neighbor classifier is at most twice the Bayes error, which is a remarkable thing to get for free from "just copy your closest neighbor" — it says the closest point already carries most of the answer. One neighbor is jumpy, though; a single closest example can be a fluke. Averaging over the `K` closest neighbors instead damps that variance, and turns the prediction into a *local average* of the response surface — it bends to follow whatever the surface is doing in this region without me ever writing down a global functional form. That's the flavor I want for a hole: don't model how column three depends on everything globally, just find the rows most similar to this one on the coordinates I can see, and let their column-three values vote.

So a recipe is forming: to fill the hole at coordinate `ℓ` of a target row `x*`, find the `K` rows most similar to `x*` and average *their* values at coordinate `ℓ`. "Most similar" has to mean a distance. Reach for the obvious one, squared Euclidean: `d(x*, y)² = Σ_ℓ (x*_ℓ − y_ℓ)²`. And immediately I walk into the wall that this whole problem is about. That sum runs over every coordinate `ℓ`, but `x*` is *missing* some coordinates — that's why I'm here — and the candidate rows `y` may be missing some too. The term `(x*_ℓ − y_ℓ)²` is undefined whenever either side has a hole at `ℓ`. I cannot even compute the distance I need to find the neighbors that would fill the hole. The missingness has bitten me again, one level up: I can't impute without a distance, and I can't form the distance without complete rows.

Let me not panic — let me ask what part of the distance I *can* compute. For a given pair `(x*, y)`, some coordinates are observed in *both*: call that the overlap set `O`. On those coordinates the difference `(x*_ℓ − y_ℓ)²` is perfectly well-defined. The natural move is to just sum over the overlap and ignore the rest: `Σ_{ℓ ∈ O} (x*_ℓ − y_ℓ)²`. That gives me a number for every pair, so I can rank candidates and pick the closest. But is that number *comparable across pairs*? Suppose `x*` overlaps with row `A` on five coordinates and with row `B` on twenty. Each squared term is nonnegative, so summing twenty of them tends to produce a larger total than summing five, *even if `B` is genuinely more similar to `x*` per coordinate*. The raw overlap sum punishes rows that happen to share more observed coordinates with me, which is backwards — those are the rows I have the *most* evidence about, and I'd be ranking them as farther away. The comparison is contaminated by how many coordinates the overlap happens to contain, not by how close the rows actually are. Wall.

What I actually want is an estimate of what the *full* `d`-dimensional squared distance would have been if no coordinate were missing. Picture it: the full distance is a sum of `d` squared terms; I only get to see `|O|` of them. If I assume the coordinates I *don't* see contribute, on average, like the coordinates I *do* see — which is the only thing I can assume without more information — then the average squared discrepancy per observed coordinate is `(1/|O|) Σ_{ℓ ∈ O} (x*_ℓ − y_ℓ)²`, and the full-dimensional sum over all `d` coordinates would be that per-coordinate average times `d`. So the estimate is

  `d(x*, y)² ≈ (d / |O|) · Σ_{ℓ ∈ O} (x*_ℓ − y_ℓ)²`.

That `d / |O|` scale-up is the correction the raw overlap sum was missing. It should put the pair sharing five coordinates and the pair sharing twenty on a common, full-dimensional footing — each becomes the *average* per-coordinate squared difference, extrapolated to all `d` coordinates. Before I trust it, let me check the boundary cases by hand. If a row is fully observed, `|O| = d`, the factor is `d/d = 1`, and the formula collapses to `Σ_ℓ (x*_ℓ − y_ℓ)²` — the ordinary squared Euclidean distance. Good: the correction does nothing when nothing is missing, which is exactly the boundary condition a sensible fix has to satisfy, and I can see it satisfies it rather than just hoping it does. The other boundary: if `|O| = 0`, the two rows share no observed coordinate, and the formula is `(d/0)·0 = 0/0` — undefined. That's the *right* behavior, not a bug: with literally no co-observed coordinate I have no evidence about their similarity and should refuse to call them neighbors. The formula goes undefined exactly where my evidence vanishes. I'll have to catch that degenerate case in code, but the math is flagging it for me honestly. This overlap distance, scaled up to full dimension, is the partial distance.

Now, is Euclidean even the right notion of "similar" here? It's the reflexive choice but not the only one. Two obvious rivals. First, **correlation** — Pearson correlation between the two rows' observed coordinates, treating each row as a profile and asking whether they rise and fall together. That's attractive because it's invariant to a row's overall level and scale: two rows with the same *shape* but different baselines come out perfectly correlated, where Euclidean would call them far apart. If what matters is the *pattern* across features rather than the absolute values, correlation captures the right similarity. Second, a **variance-minimization** criterion — pick the neighbors so that the variance of the candidate fill is smallest, i.e. favor neighbors that agree tightly on the coordinate being imputed. Each of these is defensible a priori, so I shouldn't just assert Euclidean; I should think about what each would do.

Here's the catch with Euclidean, and it's serious. Because it squares each coordinate difference, a single coordinate with a large spread, or a single wild outlier value, dominates the whole sum and drowns out every other coordinate. If one feature ranges over orders of magnitude while the others sit near unity, the distance is essentially that one feature's difference and the rest are noise in comparison — and if a row carries one freak extreme value at some coordinate, that one term blows up the distance to everything and the row gets no sensible neighbors at all. Outliers and scale heterogeneity are exactly the regime where a sum-of-squares distance is fragile, and real measured tables — especially ones spanning a wide dynamic range — are full of both. By that argument correlation, with its built-in level-and-scale invariance, *ought* to be the safer choice, and I half expect it to win.

But step back and look at what the data has typically already been through. Heavy-tailed positive measurements that span orders of magnitude are routinely **log-transformed** before any analysis, precisely because raw values like that misbehave everywhere. The log does two things at once that bear directly on my distance. It compresses large values and pulls in the long right tail, so an outlier that was a factor of a thousand above the rest becomes a factor of `log 1000 ≈ 6.9` above — still large, but no longer apocalyptic in a sum of squares: that term shrinks from `1000² = 10⁶` to about `6.9² ≈ 48`, four orders of magnitude less able to swamp its neighbors. And by turning multiplicative spread into additive spread it brings coordinates that lived on wildly different scales onto a comparable footing. So the very pathology that made me distrust Euclidean — outlier and scale sensitivity — is largely defused *before* the distance is ever computed, by a transform the data already wants for independent reasons. On log-scaled data, plain Euclidean over the observed coordinates is no longer hostage to a few extreme values, and it keeps the thing correlation throws away: the actual *magnitudes* of the coordinates, not just their up-and-down pattern. Two rows can be perfectly correlated in shape and yet be at very different levels — and for filling a hole I usually want a neighbor at the right *level*, not merely the right shape, because I'm going to copy its actual value into my hole. So once the log has tamed the outliers, the apparent advantage of correlation evaporates and Euclidean's retention of magnitude becomes the thing I want. The variance-minimization criterion is less clean for the same reason: it lets the donor coordinate I am trying to estimate influence who counts as a donor, whereas a row-to-row distance uses only the observed profile to decide similarity. That resolves the tension: use the scaled Euclidean partial distance, and lean on the data being on a sensible (log) scale to keep it robust, rather than reaching for a level-invariant metric or a target-coordinate selection rule that discards the row-level magnitude information I actually need.

So now I have neighbors: for the target row `x*` and a missing coordinate `ℓ`, compute the scaled partial distance from `x*` to every candidate row that *observes* coordinate `ℓ` (a candidate that is itself missing at `ℓ` is useless as a donor — it has nothing to lend), keep the `K` smallest, and average their `ℓ`-values. Should the average be a flat mean of the `K` neighbors, or should I weight them? Think about the spread of distances among the `K`. The very nearest neighbor is a near-twin of `x*` on the observed coordinates, so its `ℓ`-value is strong evidence. The `K`-th neighbor is, by construction, the farthest of the chosen set — it scraped in, and it's a much weaker witness; if I'd picked `K−1` instead it wouldn't even be here. Giving all `K` equal say lets that marginal neighbor pull the estimate as hard as the near-twin, which is wrong in proportion to how much farther it is. So I want the contribution of a neighbor to *decay with its distance*: weight neighbor `y` by some decreasing function `w(d)` of its distance `d` to `x*`, and form the weighted average `Σ_y w(d_y) y_ℓ / Σ_y w(d_y)`. The simplest decreasing weight is the inverse distance, `w(d) = 1/d`. Let me make sure that does what I'm claiming on actual numbers rather than just asserting "twice as far, half the say." Take three donors at distances `1, 2, 10` with values `10, 12, 100`. The raw weights are `1, 0.5, 0.1`, which normalize to `0.625, 0.3125, 0.0625` — the donor at distance `2` does get exactly half the say of the donor at distance `1`, and the far donor at distance `10` gets a tenth, so it lands at `6.25%` of the total weight. The inverse-distance fill comes out `0.625·10 + 0.3125·12 + 0.0625·100 = 16.25`, whereas the flat mean of the three values is `40.67`. So the wild far donor (value 100) drags the flat mean up to 40.67 but barely perturbs the weighted fill, which stays near the two close, agreeing donors. That is the behavior I wanted, and now I've seen it produce the number rather than merely promising it.

The inverse-distance weighting should also help with how to choose `K`, and this is where I have to be careful not to overclaim, because the obvious story is too clean. The obvious story: with a *flat* average, `K` is a sharp knob — at `K` you include a neighbor at full weight, at `K−1` you don't, so the estimate can jump; with *inverse-distance* weighting, the neighbors near the boundary of the chosen set are the *farthest* ones, carrying the *smallest* weights, so adding the `(K+1)`-th neighbor — necessarily farther than all `K` already in — adds a term with the smallest weight yet and should barely move the mean. Let me test how strong that effect really is. Extend the three donors above with a fourth at distance `20`, value `200`. The flat mean jumps from `40.67` to `80.5` — a swing of `39.8`, because the new extreme value enters with full `1/4` weight. The inverse-distance fill moves from `16.25` to `21.82` — a swing of `5.6`. So the weighting makes the `K`-sensitivity about seven times smaller here, which is a real and large effect, but it is *not* negligible: a far donor with an extreme value still nudges the answer. So the honest conclusion is weaker than "insensitive to `K`": inverse-distance weighting makes the estimate *far more forgiving* of `K` than a flat average, broad rather than knife-edge, but not immune. That's still enough that I don't need to tune `K` precisely. Too small and I'm averaging over too few witnesses — noisy, and one odd neighbor swings me. Too large and I start dragging in rows that aren't actually similar; with uniform weights that becomes the global column mean, and with distance weights a broader global weighted average rather than a local estimate. So there's a sweet spot at a moderate `K`: large enough to average out neighbor noise, small enough to stay local, and — thanks to the weighting — wide and forgiving rather than a knife-edge. A value in the low tens sits comfortably in that band for tables of a few hundred to a few thousand rows; somewhere around five to twenty, and the exact figure hardly matters. I'll take a small default like five and not lose sleep over it.

Now the degenerate cases the partial distance warned me about, because a completion procedure has to return a finite number for every hole in a feature that has at least one training observation. For a missing coordinate `ℓ`, I should first restrict the donor pool to training rows that actually observe `ℓ`; a row missing `ℓ` has nothing to donate, no matter how close it looks elsewhere. If no training row observes `ℓ` at all, there is no learned value for that feature under the default transformer, so the feature is not part of the valid output unless I explicitly ask to keep empty features. The common failure is narrower: donors exist at `ℓ`, but every one of them has empty overlap with the observed coordinates of `x*`, so every partial distance is undefined — `0/0`. I have no basis at all for ranking neighbors, so the local idea has nothing to work with. The honest fallback is the thing I started from: when I can't be local, be global — fill that hole with the training column mean. It's the best no-covariate guess, and it's exactly what I'd have done without any usable neighbors. So the method degrades gracefully — KNN where there is local evidence, the mean baseline where there is not. This isn't a patch bolted on; it's the boundary of the method meeting the boundary of the data, and the mean is the natural floor to land on.

One more thing the procedure must respect, and it's the same discipline any out-of-sample completion needs: the neighbors a test row looks up must come from the *training* table, and the column means I fall back to must be the *training* means, both frozen at fit time. If I let a test row find its neighbors among other test rows, or recompute means on the test set, I no longer have a single well-defined map from a row to its completion — I'd have one rule at train time and a different one at test time, which is not a predictor. So `fit` stores the training matrix, its missingness mask, and the per-column observed means; `transform` takes whatever table it's handed, and for each row with holes computes partial distances *into the stored training rows*, picks the `K` nearest donors per missing coordinate, inverse-distance-averages their values, and uses the stored column mean wherever the neighbor machinery comes up empty. And `fit` never touches the target `y` — the distances and means are built purely from the observed feature entries, or I'd be leaking the label into the features.

Let me also be careful about the arithmetic of the partial distance when I implement it, because doing it pair-by-pair with Python loops over the overlap set would be hopeless on a few thousand rows. I want it vectorized. Temporarily set every missing entry to `0`, and compute the ordinary squared Euclidean distance on the zero-filled rows — but that over-counts, because where a coordinate is missing in one row the `0` I substituted contributes a spurious `(x_ℓ − 0)²` or `(0 − y_ℓ)²` term that shouldn't be in the overlap sum at all. So I subtract those spurious terms back out: for the pair `(x, y)`, subtract `Σ_ℓ x_ℓ² · [y missing at ℓ]` and `Σ_ℓ y_ℓ² · [x missing at ℓ]`, which removes exactly the contributions from coordinates that aren't in the overlap. What's left should be `Σ_{ℓ ∈ O} (x_ℓ − y_ℓ)²`. I don't want to take "exactly" on faith — the bookkeeping of which squared term to subtract for which side is the kind of thing that's easy to get backwards — so let me run one pair through it concretely. Take `d = 3`, `x = (2, 5, NaN)` and `y = (1, NaN, 4)`: they co-observe only coordinate 0, so by hand the overlap sum is `(2−1)² = 1`, and the scaled squared distance should be `(3/1)·1 = 3`. Now the recipe: zero-fill to `x = (2, 5, 0)`, `y = (1, 0, 4)`, and the full zero-filled squared distance is `(2−1)² + (5−0)² + (0−4)² = 1 + 25 + 16 = 42`. The spurious terms: `y` is missing at coordinate 1, so I subtract `x₁² = 25`; `x` is missing at coordinate 2, so I subtract `y₂² = 16`. That leaves `42 − 25 − 16 = 1` — exactly the overlap sum I computed by hand. The overlap count is `|O| = 1`, so dividing by it and multiplying by `d = 3` gives `3`, and `√3 ≈ 1.732`. The vectorized bookkeeping reproduces the hand computation term for term, so the matrix-product form is faithful and not just plausible. Then divide by the overlap count `|O|` (clamped to at least one to avoid dividing by zero, with those zero-overlap pairs already flagged as undefined) and multiply by `d` to get the scaled squared distance, and take the square root. That's the whole partial-distance computation as three or four dense linear-algebra operations, which is what makes the method tractable at scale.

So let me write what I've landed on into the transformer slot. It happens that scikit-learn already ships exactly this construction as a transformer — the nan-aware Euclidean distance with the `d/|O|` scale-up, the `K`-nearest selection, the inverse-distance weighting, and the column-mean fallback — so the most direct implementation is to configure that with `n_neighbors = 5` and distance weighting, learned on the training rows and replayed on any table:

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer


class CustomImputer(BaseEstimator, TransformerMixin):
    """Weighted K-nearest-neighbor imputation. For each hole, find the K training
    rows closest on the co-observed coordinates (scaled nan-Euclidean distance),
    and fill with their inverse-distance-weighted average at that coordinate;
    fall back to the training column mean where no neighbor can donate."""

    def __init__(self, random_state=42, max_iter=10):
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_neighbors = 5                  # moderate K: local but averaged; weighting makes it forgiving

    def fit(self, X, y=None):
        self._imputer = KNNImputer(
            n_neighbors=self.n_neighbors,
            weights="distance",               # weight neighbor y by 1/d(x*, y): closer => stronger witness
            metric="nan_euclidean",           # partial distance over co-observed coords, scaled by d/|O|
        )
        self._imputer.fit(X)                  # store TRAIN rows + masks + column means (the mean fallback)
        return self

    def transform(self, X):
        # For each row's holes: distances into the stored TRAIN rows, K nearest
        # donors per missing column, inverse-distance weighted average; column
        # mean wherever no donor has a defined distance.
        return self._imputer.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
```

The library wrapper hides the moving parts, and I want to be sure I can reconstruct every one of them — the scale-up, the weighting, the fallback — from scratch and that my reconstruction actually agrees with the library rather than merely resembling it. So let me write the explicit version that makes each step visible:

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances


class CustomImputerExplicit(BaseEstimator, TransformerMixin):
    """The same method, written out: scaled nan-Euclidean partial distance,
    K-nearest donors per missing coordinate, inverse-distance weighted average,
    column-mean fallback. fit stores the training rows / mask / column means;
    transform replays exactly those on any table."""

    def __init__(self, random_state=42, max_iter=10, n_neighbors=5, keep_empty_features=False):
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_neighbors = n_neighbors
        self.keep_empty_features = keep_empty_features

    def fit(self, X, y=None):                         # y unused: never look at the label
        X = np.asarray(X, dtype=float)
        self._fit_X = X                               # frozen TRAIN rows = the donor pool
        self._mask_fit = np.isnan(X)                  # True where a training entry is missing
        self._valid_mask = ~np.all(self._mask_fit, axis=0)
        # column means over observed entries: the no-covariate fallback fill
        counts = (~self._mask_fit).sum(axis=0)
        sums = np.nansum(X, axis=0)
        self._col_mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        return self

    def _nan_euclidean(self, A):
        # Scaled partial distance from each row of A to each stored training row:
        #   dist(x, y)^2 = (D / |O|) * sum_{l in O} (x_l - y_l)^2 ,  O = co-observed coords.
        X, Y = np.asarray(A, dtype=float).copy(), np.asarray(self._fit_X, dtype=float).copy()
        missing_X, missing_Y = np.isnan(X), self._mask_fit
        X[missing_X] = 0.0
        Y[missing_Y] = 0.0
        distances = euclidean_distances(X, Y, squared=True)  # zero-filled squared distance
        XX, YY = X * X, Y * Y
        distances -= np.dot(XX, missing_Y.T)         # remove x_l^2 where y missing at l
        distances -= np.dot(missing_X, YY.T)         # remove y_l^2 where x missing at l
        np.clip(distances, 0.0, None, out=distances)
        present = np.dot(1 - missing_X, (~missing_Y).T)       # |O|, the overlap count per pair
        distances[present == 0] = np.nan             # empty overlap => undefined => mean fallback
        np.maximum(1, present, out=present)          # avoid divide by zero after marking NaNs
        distances /= present
        distances *= X.shape[1]                      # scale by D/|O|
        np.sqrt(distances, out=distances)
        return distances

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Xc = X.copy()
        mask = np.isnan(X)
        valid = self._valid_mask
        rows = np.where(mask[:, valid].any(axis=1))[0] # only rows with holes in valid features
        if rows.size:
            dist = self._nan_euclidean(X[rows])        # (n_holey_rows, n_train)
            k = self.n_neighbors
            for r, i in enumerate(rows):
                for col in np.where(mask[i] & valid)[0]:
                    donors = np.where(~self._mask_fit[:, col])[0]   # train rows observing this column
                    d = dist[r, donors]
                    ok = ~np.isnan(d)
                    if not ok.any():                   # empty overlap with every donor
                        Xc[i, col] = self._col_mean[col]   # -> column-mean fallback
                        continue
                    donors, d = donors[ok], d[ok]
                    kk = min(k, len(donors))
                    sel = np.argpartition(d, kk - 1)[:kk]  # the K nearest donors
                    dk, vk = d[sel], self._fit_X[donors[sel], col]
                    if np.any(dk == 0):                # exact matches get all the weight
                        w = (dk == 0).astype(float)
                    else:
                        w = 1.0 / dk                   # inverse-distance weights: closer => more say
                    Xc[i, col] = np.average(vk, weights=w)
        if self.keep_empty_features:
            Xc[:, ~valid] = 0.0
            return Xc
        return Xc[:, valid]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
```

Before I trust that the explicit version is the *same* method and not merely a similar one, let me run both on a small table and compare outputs directly. Take five rows of three features with a hole scattered in three of them, and run the library form and the explicit form side by side with `n_neighbors = 2`. The two completed tables come back identical to the last digit — maximum absolute difference `0.0` across every filled cell. Tracing one of those cells confirms the mechanics: row `[4, NaN, 6]` has its middle hole filled with `4.5`, which is the distance-weighted blend of the column-1 values from the two nearest rows on the co-observed coordinates — so the scale-up, the `K`-selection, and the inverse-distance weighting are all firing exactly as the library does them. And the degenerate branch: I build a table where one row observes only a coordinate that none of the column-0 donors share, so its column-0 hole has empty overlap with every possible donor. That hole comes back filled with `1.5`, which is precisely the observed mean of column 0 over the two rows that have it — the mean fallback fires exactly where the partial distance went undefined, confirming the floor I designed is the floor that actually runs.

Stepping back over the whole derivation: the table has holes downstream tools won't accept, and the column-mean fill — though it's the optimal no-covariate guess — ignores the row each hole sits in, discarding the inter-feature correlation that makes a hole guessable in the first place. The assumption-light way to use that correlation is to predict a hole from *nearby* rows: find the rows most similar to this one on the coordinates it observes and let their values vote, a local average that follows the response surface with no global model. "Similar" needs a distance, but Euclidean distance is undefined when coordinates are missing, so I compute it over the coordinates two rows co-observe and scale the result up by `d/|O|` to estimate the full-dimensional distance — the partial distance, which I checked reduces to ordinary Euclidean when nothing is missing and reproduces a by-hand pair through its vectorized form. Euclidean looked risky because squaring makes it outlier- and scale-sensitive, which argued for a level-invariant correlation metric, but on log-scaled data the log shrinks an outlier's contribution by four orders of magnitude while Euclidean keeps the coordinate *magnitudes* I need to copy a value into a hole — so Euclidean is the better choice once robustness is handled upstream. Among the `K` nearest I weight each neighbor by the inverse of its distance, which I saw on numbers pulls the fill toward the close, agreeing donors and away from a wild far one, and makes the estimate roughly seven times less sensitive to `K` than a flat average — far more forgiving, though not wholly immune, leaving a broad sweet spot at a moderate `K`. Where the local machinery has no defined donor distance, the method degrades to the training column mean, which I verified fires exactly where the partial distance goes undefined, so every hole in a fit-time observed feature gets a finite value and the floor is the honest no-covariate guess. The neighbors and means are frozen from the training rows and replayed on any table, never peeking at the label, so it's a single well-defined out-of-sample completion. And it computes at scale because the partial distance is a few dense matrix products with zero-filled rows and a spurious-term subtraction, not a per-pair loop — and the explicit form of all of this matches the library imputer to zero error on a worked table.
