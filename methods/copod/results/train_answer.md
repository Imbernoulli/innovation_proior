The problem is unsupervised outlier detection on a single batch of unlabeled tabular data: n rows and d features, with no labels at fit time, and the goal is a score per row that is larger when the row is more anomalous. A practical scoring rule needs to be cheap, robust in high dimensions, deterministic, free of hyperparameters that would require label-free tuning, and ideally interpretable about which features drove a high score. Existing methods only clear some of these bars. Proximity detectors such as LOF score by local neighbor density, but the neighbor search is O(n^2), distances concentrate as d grows, and they miss global tail events. One-class SVM estimates a support boundary, yet it requires choosing a kernel, bandwidth, and nu without labels. Histogram-based methods such as HBOS are fast and parameter-light, but they score by estimated density rather than extremeness, confuse points in low-density valleys with genuine tail outliers, and still carry a bin-count knob.

The core gap is the per-feature quantity being measured. Density confuses two different reasons a point can be rare: it can sit far out in a tail, or it can fall between two modes in a valley that is not extreme at all. What we actually want is a tail probability: how far into a tail does the value lie in its own marginal distribution? The empirical cumulative distribution function F_hat_j(x) = (1/n) sum_i I(X_{j,i} <= x) gives exactly that. By the probability integral transform, F(X) is the canonical position-in-distribution, so u_j = F_hat_j(x_j) near 0 means the value is deep in the lower tail, with no bins and no bandwidth. This is the quantity we should sum over features.

I propose COPOD, Copula-Based Outlier Detection. It keeps the useful independence skeleton of HBOS — a sum of per-feature negative-log contributions — but replaces the histogram density with empirical-CDF tail probabilities. By Sklar's theorem the joint left tail factors through a copula over the marginals. Taking the independence copula gives a product of per-feature tail probabilities, which underflows exponentially in high dimensions; taking negative logs turns that product into the ranking-preserving sum -sum_j log(u_j). The right tail is handled by the same construction on negated values, since the left ECDF of -X gives P(X >= x) cleanly and avoids the degeneracy 1 - F_hat(max) = 0 that would produce -log(0).

Because a feature's outliers are often one-sided, averaging both tails blindly would dilute the signal when the skewness tells us which side matters. COPOD therefore selects the informative tail per feature by the sign of the feature's skewness: negative skew favors the left tail, positive skew favors the right tail, and zero skew keeps both. To guard against an unreliable skewness sign, it takes, per feature, the maximum of the skew-selected contribution and the two-tail average before summing. The final score is thus a sum of nonnegative per-feature contributions, which makes it immediately interpretable: the entries of the per-feature contribution matrix reveal exactly which features made a row anomalous.

The only external number, contamination, sets only the binary inlier-outlier threshold on the already-computed scores; it does not enter the ranking. The ranking itself is fully determined by the data, deterministic, O(n log n) per feature and linear in d, and indifferent to high dimensions because the log-sum tames underflow.

```python
import numpy as np
from scipy.stats import skew as skew_sp


def skew(X, axis=0):
    return np.nan_to_num(skew_sp(X, axis=axis))


def column_ecdf(matrix):
    """Per-column empirical CDF F_hat(x) = #{X_i <= x} / n, support {1/n, ..., 1}.
    Equal values take the largest (rightmost) probability."""
    assert matrix.ndim == 2
    n, d = matrix.shape
    probabilities = np.linspace(np.ones(d) / n, np.ones(d), n)
    sort_idx = np.argsort(matrix, axis=0)
    sorted_mat = np.take_along_axis(matrix, sort_idx, axis=0)
    for c in range(d):
        for r in range(n - 2, -1, -1):
            if sorted_mat[r, c] == sorted_mat[r + 1, c]:
                probabilities[r, c] = probabilities[r + 1, c]
    reordered = np.empty_like(probabilities)
    np.put_along_axis(reordered, sort_idx, probabilities, axis=0)
    return reordered


class COPOD:
    """Copula-Based Outlier Detector. Parameter-free; score = sum over features
    of negative-log empirical-copula tail probabilities."""

    def __init__(self, contamination=0.1):
        self.contamination = contamination

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.decision_scores_ = self.decision_function(X)
        self.X_train = X
        self.threshold_ = np.quantile(self.decision_scores_, 1 - self.contamination)
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(int)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if hasattr(self, 'X_train'):
            original_size = X.shape[0]
            X = np.concatenate((self.X_train, X), axis=0)

        U_l = -1 * np.log(column_ecdf(X))    # -log P(X_j <= x)
        U_r = -1 * np.log(column_ecdf(-X))   # -log P(X_j >= x)

        skewness = np.sign(skew(X, axis=0))
        U_skew = U_l * -1 * np.sign(skewness - 1) \
            + U_r * np.sign(skewness + 1)

        O = np.maximum(U_skew, np.add(U_l, U_r) / 2)
        if hasattr(self, 'X_train'):
            scores = O.sum(axis=1)[-original_size:]
        else:
            scores = O.sum(axis=1)
        return scores.ravel()
```
