**Problem.** The frozen classifier ranks well but its probabilities lie. Learn a score-to-probability
map from the held-out calibration split, with as few shape assumptions as possible. The exact empirical
map overfits (scores are sparse), the fitted sigmoid is one rigid shape, and equal-count binning has
arbitrary boundaries and a cross-validated bin count.

**Key idea.** Trust only the one robust prior — the map is **monotone non-decreasing** — and fit the
best non-decreasing function from score to label by order-constrained least squares,
`min Σ_i w_i (g_i − ĝ_i)²` s.t. `ĝ` non-decreasing, solved by **pool-adjacent-violators**. The optimum
is piecewise constant, each block equal to the empirical positive rate of its examples (so each value is
an honest probability in `[0,1]`), and the blocks are a *data-chosen* binning — coarse where the ranking
is poor, fine where it's good. No shape assumed, no bin count to tune.

**Why it works.** Monotonicity is weak enough to fit any monotone distortion (sigmoidal or not) yet
strong enough to forbid wiggling. Squared error makes each constant piece the empirical rate; the same
fit holds under log-loss (both are Bregman losses for the mean).

**Scaffold edit / hyperparameters.** Binary: one `IsotonicRegression(out_of_bounds="clip", y_min=0,
y_max=1)` on the positive-class scores vs 0/1 labels (the class *is* PAV — sort, tied-score averaging,
threshold trimming, linear interpolation, input-clamp-to-domain, all inside it). Multiclass: refuse the
joint simplex (no 1-D order, curse of dimensionality); do **one-against-all** per class, then floor at
`1e-15` and renormalize so rows sum to 1.

**What to watch.** Non-parametric ⇒ more degrees of freedom ⇒ needs more calibration data. Expect strong
**ECE** (the flat, coarse map matches accuracy bin-by-bin, especially on the small binary tasks) but
relatively weak **NLL/Brier** (the proper scores punish a coarse map for handing the same block value to
examples that deserve different probabilities — worst where the calibration split is small and the curve
steep). That split is what forces a parametric, data-efficient family next.

```python
# EDITABLE region of custom_calibration.py (lines 45-102) - step 1: isotonic regression
class CalibrationMethod(BaseEstimator):
    """Isotonic Regression calibration.

    Fits a non-parametric, monotonically non-decreasing function
    from uncalibrated probabilities to calibrated ones.
    """

    def __init__(self):
        self.is_binary = None
        self.calibrators_ = None

    def fit(self, probs, labels):
        from sklearn.isotonic import IsotonicRegression as IR

        if probs.ndim == 1:
            self.is_binary = True
            iso = IR(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(probs, labels)
            self.calibrators_ = [iso]
        else:
            self.is_binary = False
            n_classes = probs.shape[1]
            self.calibrators_ = []
            for c in range(n_classes):
                binary_labels = (labels == c).astype(float)       # one-against-all per class
                iso = IR(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(probs[:, c], binary_labels)
                self.calibrators_.append(iso)
        return self

    def predict_proba(self, probs):
        if self.is_binary:
            calibrated = self.calibrators_[0].predict(probs)
            return np.clip(calibrated, 0, 1)
        else:
            n_classes = probs.shape[1]
            calibrated = np.zeros_like(probs)
            for c in range(n_classes):
                calibrated[:, c] = self.calibrators_[c].predict(probs[:, c])   # r_c(x)
            calibrated = np.clip(calibrated, 1e-15, None)         # avoid 0/0 all-zero row
            calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)     # renormalize
            return calibrated
```
