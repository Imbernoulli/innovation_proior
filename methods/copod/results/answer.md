# COPOD, distilled

COPOD (Copula-Based Outlier Detection) is a parameter-free, deterministic, fast
unsupervised outlier detector for tabular data. It scores each row by the negative log of an
empirical-copula **tail probability**, summed over features: per feature it measures how far
into a tail the value sits using an empirical CDF (no bins, no bandwidth), picks the informative
tail by the feature's skewness, and adds the per-feature contributions. Higher score = more
anomalous. It is the empirical-tail-probability successor to the histogram score HBOS: same
sum-of-negative-logs independence skeleton, but the per-feature quantity is *extremeness* (a
CDF tail probability) instead of *density* (a histogram bar), which removes both the
density-vs-extremeness confusion and the bin-count hyperparameter.

## Problem it solves

Score a single batch of unlabeled, standardized tabular rows (`n` rows, `d` features) by how
anomalous each is, with no labels at fit time, such that the detector is cheap (near-linear in
`n` and `d`), robust in high dimensions, fully deterministic, free of hyperparameters that
would need label-free tuning, and interpretable (which features made a row anomalous).

## Key idea

1. **Tail probability, not density.** For feature `j`, the empirical CDF
   `F_hat_j(x) = (1/n) sum_i I(X_{j,i} <= x)` gives `u_j = F_hat_j(x_j)`, the probability of
   seeing something as small along feature `j`. By the probability integral transform `F(X)` is
   the canonical position-in-distribution, so `u_j` near 0 = far in the lower tail. No bins.
2. **Copula assembly + log-sum.** By Sklar's theorem the joint left tail is
   `F(x) = C(F_1(x_1), ..., F_d(x_d))`. With the independence (product) copula,
   `C(u) = prod_j u_j`, and this product underflows exponentially in `d`. Taking logs
   (monotone, ranking-preserving) turns it into a sum:
   `-log C(u) = - sum_j log(u_j)` — the sum, over features, of per-feature negative-log tail
   probabilities. Same trick as a log-likelihood / HBOS.
3. **Both tails.** A large-side outlier has `u_j ≈ 1`, so the left tail misses it. The
   continuous right tail is the survival event, linked by
   `P(X_1>x_1, ..., X_d>x_d) = C_bar(F_bar_1(x_1), ..., F_bar_d(x_d))` with `F_bar = 1 - F`.
   Computing `1 - F_hat` directly hits the empirical-support boundary
   `F_hat(max) = 1 -> 1 - 1 = 0 -> -log(0) = inf`. The implementation dodges it by computing
   the **left-tail ECDF on the negated data**: `F_hat_{-X}(-x)` gives
   `(1/n) sum_i I(X_i >= x)` cleanly, ties included.
4. **Skewness-corrected tail selection.** Averaging both tails dilutes detection when a
   feature's outliers are one-sided. The implementation works in negative-log contribution
   space: `U_l=-log(ecdf(X))`, `U_r=-log(ecdf(-X))`. With `s=sign(skew(X))`, the branch-free
   skewness term is `U_skew = U_l * -1 * sign(s - 1) + U_r * sign(s + 1)`, so `s=-1 -> U_l`,
   `s=+1 -> U_r`, and `s=0 -> U_l + U_r`. It then guards the skew sign with a two-tail average
   safety net by taking, **per feature**, `max(U_skew, (U_l + U_r)/2)`, then summing over
   features.
5. **Interpretable + parameter-free.** The score is a sum of nonnegative per-feature
   contributions, so the per-feature terms reveal which features drove the anomaly (compare to a
   band such as `-log(0.01) = 4.61`). The only external number, `contamination`, sets the
   binary label threshold (top fraction flagged); it does **not** enter the score ranking.

## Final algorithm

Inputs: data `X = (X_{1,i}, ..., X_{d,i}), i = 1..n`. Output: scores `O(X)`, higher = more
anomalous.

```
F_hat_j(x)       = (1/n) sum_i I(X_{j,i} <= x)           # left ECDF, support {1/n,...,1}
F_hat_{-X,j}(-x) = (1/n) sum_i I(-X_{j,i} <= -x)         # right tail via left ECDF of -X

U_l    = -log(ecdf(X))                                   # left-tail contributions, n by d
U_r    = -log(ecdf(-X))                                  # right-tail contributions, n by d
s      = sign(skew(X, axis=0))                           # one sign per feature
U_skew = U_l * -1 * sign(s - 1) + U_r * sign(s + 1)      # -1->U_l, +1->U_r, 0->U_l+U_r
O_dim  = max(U_skew, (U_l + U_r) / 2)                    # elementwise per-feature max
O(X)   = sum_j O_dim[:, j]                               # sum after the per-feature max
```

Note the order: the per-feature **max** is taken before the sum over features (each feature
independently uses whichever of its targeted tail or its two-tail average is most extreme).

## Relation to prior methods

- **HBOS** (Goldstein & Dengel 2012) = the same sum-of-negative-logs independence model, but
  the per-feature contribution is `-log(histogram density)` instead of `-log(tail probability)`,
  and it carries a bin-count knob. COPOD swaps density for the bin-free ECDF tail probability,
  which measures extremeness and removes the hyperparameter.
- **Copula theory** (Sklar 1959; Nelsen 2006) supplies the joint-from-marginals decomposition;
  the empirical copula makes it nonparametric. The product (independence) copula gives the
  log-sum; the survival copula gives the right tail.

## Working code

Faithful to the canonical implementation (pyod.models.copod): per-feature negative-log left and
right tail contributions, branch-free skewness selection, per-feature max against the two-tail
average, then sum over features. Fills the `decision_function` slot of the detector harness.

```python
import numpy as np
from scipy.stats import skew as skew_sp


def skew(X, axis=0):
    # per-feature skewness; nan_to_num guards constant (zero-variance) columns
    return np.nan_to_num(skew_sp(X, axis=axis))


def column_ecdf(matrix):
    """Per-column empirical CDF F_hat(x) = #{X_i <= x} / n, support {1/n, ..., 1}.
    Equal values take the largest (rightmost) probability, matching the <= count."""
    assert matrix.ndim == 2, 'matrix must be 2D for the ECDF computation.'
    n, d = matrix.shape
    probabilities = np.linspace(np.ones(d) / n, np.ones(d), n)  # 1/n .. 1 per column
    sort_idx = np.argsort(matrix, axis=0)
    sorted_mat = np.take_along_axis(matrix, sort_idx, axis=0)
    for c in range(d):                          # propagate higher prob across ties
        for r in range(n - 2, -1, -1):
            if sorted_mat[r, c] == sorted_mat[r + 1, c]:
                probabilities[r, c] = probabilities[r + 1, c]
    reordered = np.empty_like(probabilities)    # undo the sort
    np.put_along_axis(reordered, sort_idx, probabilities, axis=0)
    return reordered


class COPOD:
    """Copula-Based Outlier Detector. Parameter-free; score = sum over features
    of negative-log empirical-copula tail probabilities."""

    def __init__(self, contamination=0.1):
        self.contamination = contamination      # label threshold only; not in the score

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.decision_scores_ = self.decision_function(X)
        self.X_train = X
        self.threshold_ = np.quantile(self.decision_scores_, 1 - self.contamination)
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(int)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        # score new rows against the established training distribution
        if hasattr(self, 'X_train'):
            original_size = X.shape[0]
            X = np.concatenate((self.X_train, X), axis=0)

        # per-feature negative-log tail contributions
        self.U_l = -1 * np.log(column_ecdf(X))    # left tail  -log P(X_j <= x)
        self.U_r = -1 * np.log(column_ecdf(-X))   # right tail -log P(X_j >= x), via -X

        # skewness-selected tail (branch-free):
        #   sign<0 -> U_l ; sign>0 -> U_r ; sign==0 -> U_l + U_r
        skewness = np.sign(skew(X, axis=0))
        self.U_skew = self.U_l * -1 * np.sign(skewness - 1) \
            + self.U_r * np.sign(skewness + 1)

        # per feature: max(skew-targeted tail, two-tail average), then sum over features
        self.O = np.maximum(self.U_skew, np.add(self.U_l, self.U_r) / 2)
        if hasattr(self, 'X_train'):
            scores = self.O.sum(axis=1)[-original_size:]
        else:
            scores = self.O.sum(axis=1)
        return scores.ravel()
```

The per-feature contribution matrix `O` also gives the interpretability ("Dimensional Outlier
Graph"): `O[i, :]` is the per-feature breakdown of row `i`'s score, and features whose
contribution exceeds a chosen band (e.g. `-log(0.01) = 4.61`, or a high percentile of the
column) are the ones that made the row an outlier.
