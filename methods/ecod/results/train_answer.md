We need an unsupervised outlier score for a standardized tabular matrix with no labels, and the constraints are tight: the detector must scale to large n and d, it must have no hyperparameters at all because there is no validation signal to tune them, and it should be able to explain which features made a point look strange. Existing ideas each fail at least one of these. Density-based methods like LOF and kNN rely on pairwise distances and a neighbor count k, so they collapse under the curse of dimensionality and leave an untunable knob. One-Class SVM adds kernel and bandwidth choices plus a nu parameter and scales superlinearly. Isolation Forest is faster but still needs the number of trees and subsample size, and its score is an opaque ensemble path length. Histogram-based methods such as HBOS are linear and per-feature, but the histogram bin width is a hyperparameter and discretization creates edge artifacts. Even the simple three-sigma and 1.5·IQR rules are tuning-free and interpretable, yet they summarize each feature by only two numbers and implicitly assume a symmetric, nearly Gaussian shape, so they mis-rank skewed or heavy-tailed tails. What is missing is a way to use the whole marginal distribution of each feature, nonparametrically and without knobs, and then combine the features cleanly.

The method I propose is ECOD, short for Empirical-Cumulative-distribution-based Outlier Detection. It starts from the rare-event definition of an outlier: a point is anomalous if it lies in a low-probability region of the data distribution. The natural one-dimensional measure of tail extremeness is cumulative tail probability. For the left side this is F(z) = P(X ≤ z), estimated nonparametrically by the empirical CDF, the fraction of samples at or below z. The ECDF is an unusually good estimator here: Glivenko-Cantelli gives uniform consistency, and the Dvoretzky-Kiefer-Wolfowitz inequality gives a finite-sample rate that depends only on n and epsilon, not on the distribution or on dimension. That dimension-free guarantee holds per feature, but it does not hold for the joint ECDF, which suffers from the curse of dimensionality. So ECOD assumes features are independent, factors the joint tail into a product of univariate tails, and estimates each marginal with its own ECDF. The independence assumption trades some sensitivity to joint-structure outliers for the ability to scale linearly in n and d, to stay completely parameter-free, and to attribute the score feature by feature.

For each dimension ECOD builds two tail arrays. The left-tail ECDF counts the fraction of training values less than or equal to a query; ties take the largest rank so the value at the maximum is 1 and the log stays finite. The right tail is not computed as one minus the left ECDF, because that would use a strict inequality and give probability zero at the maximum, producing log zero. Instead ECOD computes the right-tail ECDF directly as the left ECDF of the negated column, which assigns 1/n to the maximum and preserves symmetry. The product of d tail probabilities underflows quickly, so ECOD works in negative-log space: the joint left-tail score is the sum over dimensions of -log F_hat_left, and likewise for the right tail. This additive form is numerically stable, monotone in rarity, and immediately interpretable because each term is the contribution of one feature.

A one-sided score is fragile: left-only misses outliers that are large, and right-only misses outliers that are small. Averaging the two tails dilutes a genuine one-sided signal. ECOD therefore selects the outlying tail per dimension by the sign of the feature's skewness: a negative skew coefficient means the long tail is on the left, so the left-tail probability is used; a positive skew means the long tail is on the right, so the right-tail probability is used. Skewness is cheap and has no hyperparameters. Because the skewness sign can be noisy or the marginal can be two-sided, ECOD keeps three views in play: the left-only aggregate, the right-only aggregate, and the skewness-corrected aggregate. The final score is the most extreme of the three, so a genuine one-sided outlier is never hidden by a bad skewness call. The canonical implementation takes the per-dimension maximum of the three tail views and then sums over dimensions, which is a slightly stronger aggregation than the mathematical max-of-three-sums form but uses exactly the same ingredients and remains parameter-free.

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
