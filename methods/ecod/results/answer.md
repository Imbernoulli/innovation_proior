# ECOD, distilled

ECOD (Empirical-Cumulative-distribution-based Outlier Detection) is a parameter-free
unsupervised outlier detector for tabular data. It treats outliers as rare *tail events*:
per dimension it estimates the left- and right-tail probabilities with the univariate empirical
CDF, aggregates them across dimensions under an independence assumption by summing per-dimension
negative log probabilities (so the joint tail probability is computed stably in log space), and
selects per dimension which tail is the outlying one using the sign of that dimension's
skewness. The aggregate mathematical score is the most extreme of the left-only, right-only, and
skewness-corrected aggregate sums; the canonical PyOD implementation uses the same ingredients
but takes the maximum per dimension before summing. It has no hyperparameters, parallelizes
trivially across dimensions, and is interpretable because the score decomposes additively into
per-dimension contributions.

## Problem it solves

Unsupervised outlier scoring on a standardized tabular matrix `X ∈ R^{n×d}` with no labels,
where the method must (1) scale to large `n` and `d` without density estimation or pairwise
distances (curse of dimensionality), (2) have *no* hyperparameters (tuning is impossible
without a validation signal), and (3) attribute its score to individual features.

## Key idea

Outlier = rare event = tail event. For a point `x`, use the joint left-tail event
`P(X^{(1)} ≤ x^{(1)}, ..., X^{(d)} ≤ x^{(d)})` and the mirror right-tail event
`P(X^{(1)} ≥ x^{(1)}, ..., X^{(d)} ≥ x^{(d)})`; if either is tiny, `x` is rare. Estimating the
joint CDF directly is cursed by dimension and the joint ECDF converges slowly in `d`. So assume
the features are independent, factor the joint tail into marginal tails, and estimate each
univariate marginal CDF by its ECDF — which is consistent (Glivenko–Cantelli), finite-sample
tight at a *dimension-free* rate in 1-D (DKW: `P(sup|F_n − F|>ε) ≤ 2e^{−2nε²}`), and has
nothing to tune. Aggregate in negative-log space (the product of `d` tail probabilities
underflows; `−Σ_j log p_j` is stable, monotone, and additively per-dimension interpretable).
Pick the outlying tail per dimension by the sign of the skewness coefficient (negative skew ⇒
long left tail ⇒ score the left tail; positive ⇒ right). Keep left-only, right-only, and
skewness-corrected views so a one-sided outlier is not hidden by a bad skew call.

## Final algorithm

For each dimension `j` (with `X̄^{(j)}` the column mean):

- Left-tail ECDF:  `F̂_left^{(j)}(z)  = (1/n) Σ_i 1{X_i^{(j)} ≤ z}` (rank-based; ties take the
  largest rank, so it equals `P(X ≤ z)`; values in `{1/n,…,1}`, never 0).
- Right-tail ECDF: `F̂_right^{(j)}(z) = (1/n) Σ_i 1{X_i^{(j)} ≥ z}` (computed as the left ECDF
  of `−X` at `−z`; the maximum value gets `1/n`, log-safe — not `1 − F̂_left`, which would be 0
  there and uses a strict inequality, breaking the left/right symmetry).
- Skewness coefficient:
  `γ_j = [ (1/n) Σ_i (X_i^{(j)} − X̄^{(j)})³ ] / [ (1/(n−1)) Σ_i (X_i^{(j)} − X̄^{(j)})² ]^{3/2}`.

For each point `X_i`, the aggregate mathematical form uses three scores:

```
O_left(X_i)  = − Σ_j log F̂_left^{(j)}(X_i^{(j)})
O_right(X_i) = − Σ_j log F̂_right^{(j)}(X_i^{(j)})
O_auto(X_i)  = − Σ_j [ 1{γ_j < 0} log F̂_left^{(j)}(X_i^{(j)})
                     + 1{γ_j ≥ 0} log F̂_right^{(j)}(X_i^{(j)}) ]
O_i          = max{ O_left(X_i), O_right(X_i), O_auto(X_i) }      # most extreme of the three
```

Return `O = (O_1, …, O_n)`; higher = more anomalous. The indicator version breaks `γ_j = 0`
toward the right tail; this is a measure-zero convention for continuous nonconstant data.

The canonical PyOD implementation realizes the skew choice branch-free. With
`U_l = −log F̂_left`, `U_r = −log F̂_right`, and `s_j = sign(γ_j)`, it computes
`U_skew = U_l * (−sign(s_j − 1)) + U_r * sign(s_j + 1)`, which decodes to `s_j = −1 -> U_l`,
`s_j = +1 -> U_r`, and `s_j = 0 -> U_l + U_r`. It then returns
`Σ_j max{U_l^{(j)}, U_r^{(j)}, U_skew^{(j)}}`, while the mathematical specification above uses
`max` over the three aggregate sums.

## Why each choice

- **Tail probability (CDF), not density.** "Outlier = tail event" wants cumulative tail mass,
  which is the *integral* of the density; the ECDF estimates it with no bandwidth/bins, unlike
  KDE or histograms (the latter is exactly HBOS, `Σ_j −log hist_j(x)`, with a bin-width knob).
- **ECDF, not a parametric fit.** Glivenko–Cantelli/DKW give consistency and a dimension-free
  1-D rate with no assumption on shape and nothing to tune; three-sigma/IQR use only mean+sd or
  quartiles and so assume symmetry/Gaussianity.
- **Independence factorization.** The joint ECDF is cursed by dimension; the univariate ECDF is
  dimension-free. Trade dependence-sensitivity for `d` clean marginals, `O(nd)`, parallel.
  (Under Sklar's theorem this is just taking the copula to be the independence copula.)
- **Negative-log sum.** Product of `d` probabilities underflows; `−Σ log` is stable, monotone,
  and additively attributable per dimension (free interpretability).
- **Separate `≥`-ECDF for the right tail.** `1 − F̂_left` uses a strict inequality and is 0 at
  the maximum value (`log 0 = −∞`); `P(X ≥ z)` is symmetric and assigns `1/n` there.
- **Skewness-sign tail selection.** Left-only fails on right outliers and vice versa; averaging
  the two tails dilutes a one-sided signal; brute-forcing `2^d` tail combinations is
  infeasible. The skew sign picks the long/outlying tail per dimension with no tuning.
- **`max` of the three scores.** The skewness estimate is noisy and can mis-pick a tail, or the
  marginal can be two-sided; keeping `O_left` and `O_right` in the running guarantees a genuine
  one-sided outlier is never suppressed by a wrong skew call. Still parameter-free.

## Complexity and interpretability

The scoring algebra after ECDF values are available is `O(nd)` time and space: skewness is one
pass and scoring is a sum over `d` terms. A sort-based rank implementation of the ECDF pays
`O(d n log n)` to materialize the column ECDF values, then the remaining reductions are
`O(nd)`. Dimensions are independent so the work parallelizes across columns and distributed
workers. Interpretability: because the score is an additive sum of per-dimension terms,
`O^{(j)}(X_i) = −log F̂^{(j)}(X_i^{(j)})` (the chosen tail) is the contribution of dimension
`j`, which can be ranked, plotted, and compared against a reference band such as
`−log(0.01) ≈ 4.6` to explain *which* features made a point anomalous.

## Working code

The code follows `pyod.models.ecod.ECOD` and `pyod.utils.stat_models.column_ecdf`. The only
nontrivial primitive is the rank-based column ECDF with correct tie handling; the detector code
uses the PyOD aggregation order.

```python
import numpy as np
from scipy.stats import skew as skew_sp


def column_ecdf(matrix):
    """Per-column ECDF: for value z in column j, return (#{X_ij <= z}) / n.
    Ties take the largest rank, so F̂_left(z) = P(X <= z) exactly; values in {1/n,...,1}."""
    assert matrix.ndim == 2, "ECDF expects a 2D (n_samples x n_features) matrix"
    n = matrix.shape[0]
    # probability for sorted rank r (1-indexed) is r/n, per column
    probabilities = np.linspace(np.ones(matrix.shape[1]) / n, np.ones(matrix.shape[1]), n)
    sort_idx = np.argsort(matrix, axis=0)
    matrix = np.take_along_axis(matrix, sort_idx, axis=0)          # sort each column ascending
    # equal values all take the largest-rank probability (so it is P(X <= z))
    for cx in range(probabilities.shape[1]):
        for rx in range(probabilities.shape[0] - 2, -1, -1):
            if matrix[rx, cx] == matrix[rx + 1, cx]:
                probabilities[rx, cx] = probabilities[rx + 1, cx]
    reordered = np.ones_like(probabilities)                        # undo the sort
    np.put_along_axis(reordered, sort_idx, probabilities, axis=0)
    return reordered


def skew(X, axis=0):
    return np.nan_to_num(skew_sp(X, axis=axis))


class ECOD:
    """Empirical-CDF-based outlier detection. Parameter-free; fit/decision_function API.
    Outlier scores are comparative (not probabilities); higher = more anomalous."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.decision_scores_ = self.decision_function(X)
        self.X_train = X
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if hasattr(self, "X_train"):
            original_size = X.shape[0]
            X = np.concatenate((self.X_train, X), axis=0)

        # per-dimension negative-log tail probabilities (n x d)
        U_l = -1.0 * np.log(column_ecdf(X))      # −log F̂_left  (left-tail rarity per dim)
        U_r = -1.0 * np.log(column_ecdf(-X))     # −log F̂_right (right-tail rarity per dim)

        # skewness sign per dim: −1 long left tail, +1 long right tail
        s = np.sign(skew(X, axis=0))
        # skewness-corrected pick (branch-free): s=−1 -> U_l ; s=+1 -> U_r ; s=0 -> U_l + U_r
        U_skew = U_l * -1.0 * np.sign(s - 1) + U_r * np.sign(s + 1)

        # PyOD aggregation: per-dimension max of the three tail views, then sum over dimensions.
        O = np.maximum(np.maximum(U_l, U_r), U_skew).sum(axis=1)
        if hasattr(self, "X_train"):
            O = O[-original_size:]
        return O.ravel()
```

Implementation notes faithful to the detector: (1) the code above aggregates by taking the
elementwise per-dimension `max(U_l, U_r, U_skew)` and then summing over dimensions, whereas the
mathematical specification takes the `max` of the three total sums; both are parameter-free
realizations, differing only in the order of `max` and sum. (2) `decision_function` is
transductive: when a fitted detector scores new points it concatenates the training and test
rows before computing the ECDFs, so the empirical tails are estimated over the pooled data. (3)
SciPy's skewness routine is used only through its sign; the coefficient above and the
implementation's standardized third moment have the same sign for nonconstant columns. (4) A
dimensional-outlier plot of `O^{(j)}(X_i)` against a percentile band surfaces the per-feature
explanation.
