**Problem.** The global scalar bottomed out: temperature scaling left Adult/COMPAS worst-group ECE near
0.484/0.485 because one degree of freedom can soften uniformly but cannot *bend*. If the true
score→probability distortion is non-uniform or asymmetric, a single slope leaves residual
miscalibration on the worst subgroup. I want a richer, still-monotone, still group-agnostic global map.

**Key idea.** Isotonic regression: fit the best *non-decreasing* map from raw positive-class score to
the 0/1 label by order-constrained least squares, solved by pool-adjacent-violators (PAVA). The only
prior is monotonicity (the classifier ranks well, confirmed by `subgroup_auroc`); within that class the
curve bends however the calibration data wants. Each PAVA block's value is the weighted mean of its 0/1
targets — the empirical positive rate — so every constant piece is an honest probability, and squared
error and log-loss give the identical fit (both Bregman losses for the mean).

**Why.** The fitted intervals are a *data-chosen binning* — coarse where the classifier ranks poorly,
fine where it ranks well — which repairs equal-count binning's arbitrariness with no bin count to
cross-validate. Out-of-range test scores (likely under the shift) are clipped to the fitted score domain;
fitted values are bounded to `[0,1]`. The cost: non-parametric, so it needs more calibration data and can
overfit a small shifted split. The task is binary end to end (the harness always passes a 1-D `probs`),
so one isotonic map suffices — no multiclass one-against-all reconciliation. `groups` is ignored.

**Hyperparameters.** `IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")` — no knobs to
tune (PAVA chooses the block structure); `eps = 1e-6` clip on the output for valid probabilities.

```python
class CalibrationMethod:
    """Isotonic regression calibration."""

    def __init__(self):
        self.eps = 1e-6
        self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, probs, labels, groups=None):
        probs = np.asarray(probs).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)
        self.model_.fit(probs, labels)
        return self

    def predict_proba(self, probs, groups=None):
        probs = np.asarray(probs).reshape(-1)
        return np.clip(self.model_.predict(probs), self.eps, 1.0 - self.eps)
```
