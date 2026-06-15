**Problem.** Temperature scaling is too rigid (one slope, transfers under shift but cannot bend, leaving
Adult worst-group ECE at 0.484); isotonic is too flexible (bends to fix Adult at 0.348 but overfits the
small shifted Law School split up to 0.451 and nicks the ranking). I want the map in between: able to
bend — including the *gathering* shapes a sigmoid cannot make — but inside a tightly parametric family
whose few parameters transfer under shift.

**Key idea.** Beta calibration: assume the per-class scores are **beta-distributed** (the natural density
on `[0,1]`) instead of the equal-variance Gaussians behind the sigmoid. The likelihood ratio of two
betas is a power law on the odds, `e^c · s^a/(1−s)^b`, whose calibrated posterior
`μ_beta(s;a,b,c) = 1/(1 + 1/(e^c·s^a/(1−s)^b))` is a three-parameter family containing sigmoids
(`a=b>1`), inverse sigmoids (`a=b<1`, which pull extreme over-confident scores back), asymmetric maps
(`a≠b`), and the identity (`a=b=1,c=0`). Because `ln LR` is linear in `log s` and `log(1−s)`, the whole
family is fitted by a **bivariate logistic regression** on those two features — one off-the-shelf call,
the cost of a sigmoid, with three parameters instead of isotonic's many.

**Why.** Three parameters give the shape freedom that fixes Adult-style non-uniform distortion while
keeping variance far below isotonic's, so it should transfer to the shifted tail. Beta with `a=b≈1`
degenerates toward the global scale correction, so it cannot overfit the thin Law School split the way
isotonic did; and being strictly monotone (`a,b≥0`), it preserves the ranking (`subgroup_auroc`) that
PAVA's flat blocks damaged. `groups` is ignored — the group-agnostic posture that won every comparison.

**Hyperparameters / same-named-vs-paper notes.** The scaffold builds features `[log p, log1p(−p)]` —
the second feature is `+log(1−p)`, so the logistic regression learns the coefficient `−b`; this fits the
*identical* `μ_beta` as the negated-feature `(log s, −log(1−s))` convention (sign is cosmetic). It uses
default `LogisticRegression(max_iter=2000, solver="lbfgs")` — i.e. **default L2 (`C=1.0`)**, a mild
regularizer rather than the near-zero-penalty pure MLE; here that acts as variance control matched to the
shift, pulling `a,b` toward a safe constant on thin splits. It omits the canonical monotonicity
drop-and-refit guard, which rarely matters because real distortions are monotone and the unconstrained
fit returns `a,b≥0` on its own. `eps = 1e-6` clips `p` so the logs stay finite.

```python
class CalibrationMethod:
    """Beta calibration via logistic regression on transformed probabilities."""

    def __init__(self):
        self.eps = 1e-6
        self.model_ = LogisticRegression(max_iter=2000, solver="lbfgs")

    def _featurize(self, probs):
        probs = np.asarray(probs).reshape(-1)
        p = np.clip(probs, self.eps, 1.0 - self.eps)
        return np.column_stack([np.log(p), np.log1p(-p)])

    def fit(self, probs, labels, groups=None):
        X = self._featurize(probs)
        labels = np.asarray(labels).reshape(-1).astype(int)
        self.model_.fit(X, labels)
        return self

    def predict_proba(self, probs, groups=None):
        X = self._featurize(probs)
        return np.clip(self.model_.predict_proba(X)[:, 1], self.eps, 1.0 - self.eps)
```
