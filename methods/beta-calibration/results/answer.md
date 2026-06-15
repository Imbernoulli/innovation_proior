# Beta calibration, distilled

Beta calibration is a post-hoc calibration family for binary classifiers, derived by assuming
the classifier's per-class scores follow a **beta distribution** (the natural density on the
`[0, 1]` interval a probability-like score lives on) rather than the equal-variance Gaussian
that underlies logistic / Platt scaling. The likelihood ratio of two betas is a power law on
the odds, `e^c * s^a / (1-s)^b`, whose calibrated posterior is a three-parameter family that
contains sigmoids, inverse sigmoids, the identity, and asymmetric maps. Because the
log-likelihood-ratio is linear in `log s` and `-log(1-s)`, the family is fitted by a single
bivariate logistic regression on those two features — exactly as cheap as fitting a sigmoid.

## Problem it solves

Mapping a classifier's raw score `s = f(x) in [0, 1]` to a calibrated posterior probability,
fitted on a small held-out calibration set, with a parametric family flexible enough to undo
real score distortions in **both** directions (scores pushed to the extremes *and* scores
pulled to the middle) and to leave an already-calibrated classifier alone.

## Key idea

Calibration via the per-class likelihood ratio: posit `p(s | +)` and `p(s | -)`, form
`LR(s) = p(s|+)/p(s|-)`, and (uniform prior) read off `mu(s) = 1/(1 + LR(s)^{-1})`. Logistic
calibration takes the two densities to be **equal-variance Gaussians**, giving
`LR(s) = exp(gamma(s - m))` and the sigmoid `mu(s) = 1/(1 + exp(-gamma(s - m)))`. That
assumption fails three ways: Gaussians put mass outside `[0, 1]`; the sigmoid can only spread
scores toward the extremes (`gamma >= 0`), never gather them, so it fits over-confident
classifiers (Naive Bayes, original Adaboost) badly; and the identity map is not in the family,
so it uncalibrates a calibrated model.

Replace the Gaussian with the **beta distribution** `p(s; alpha, beta) = s^{alpha-1}(1-s)^{beta-1}/B(alpha, beta)`.
Positives `~ Beta(alpha_1, beta_1)`, negatives `~ Beta(alpha_0, beta_0)`:

```
K = B(alpha_1,beta_1)/B(alpha_0,beta_0),
LR(s) = s^a / ((1-s)^b K) = e^c * s^a / (1-s)^b,
a = alpha_1 - alpha_0,  b = beta_0 - beta_1,  K = e^{-c},
```

giving the **beta calibration map**

```
mu_beta(s; a, b, c) = 1 / ( 1 + 1/( e^c * s^a / (1-s)^b ) ).
```

- Monotone non-decreasing iff `a, b >= 0` (since `d/ds ln LR = a/s + b/(1-s)`).
- Every such map corresponds to two real beta densities (e.g. `alpha_0=1, alpha_1=1+a, beta_0=M+b, beta_1=M`, choosing `M` to hit `e^{-c}`).
- Shapes: `a = b > 1` sigmoid; `a = b < 1` inverse sigmoid (corrects extreme scores);
  `a = b = 1, c = 0` **identity**; `a != b` asymmetric. Not translation-invariant in `s`.
- Midpoint `m` (where `mu = 1/2`, i.e. `LR(m) = 1`): `c = b ln(1-m) - a ln m`.
- Worked example: Naive Bayes on `k` identical copies of a calibrated feature outputs
  `s = x^k/(x^k+(1-x)^k)`; the exact recovering map is beta with `a = b = 1/k, c = 0`.

## Fitting (the "easily implemented" result)

**Proposition 2.** `mu_beta(s; a, b, c) = mu_bilogistic(ln s, -ln(1-s); a, b, c)`, where
`mu_bilogistic(s', s''; a, b, c) = 1/(1 + 1/exp[a s' + b s'' + c])` is a bivariate logistic
regression. *Proof:* `exp[a ln s - b ln(1-s) + c] = e^c s^a/(1-s)^b = LR_beta`. So the full
three-parameter map is fitted by **bivariate logistic regression** on features
`(ln s, -ln(1-s))` against the labels — the log-loss / MLE optima coincide because it is the
same model and objective.

**Proposition 1.** For `a = b`: `mu_beta(s; a, a, c) = mu_logistic(ln(s/(1-s)); a, c)`, so the
symmetric variant (beta[a=b]) is **univariate logistic regression** on the log-odds
`ln(s/(1-s))`. (This retroactively justifies the older linear-in-log-odds recalibration
heuristic as beta calibration with `a = b`.)

Algorithm (full 3-parameter version):

```
Require: calibration labels y_train, scores s_train; test scores s_test
1: s'  <- ln(s_train)
2: s'' <- -ln(1 - s_train)
3: (a, b, c) <- fit bivariate logistic regression predicting y_train from (s', s'')
4: p_hat_test <- 1 / (1 + 1/( e^c * s_test^a / (1 - s_test)^b ))     # == sigmoid(a ln s_test - b ln(1-s_test) + c)
5: return p_hat_test
```

Subtlety: unconstrained logistic regression could in principle return `a < 0` or `b < 0`
(non-monotone). In practice it rarely does, since real distortions are monotone; if a
guarantee is needed, fit with `a, b >= 0`, or fit unconstrained, and if a coefficient is
negative drop that feature (fix it to 0) and refit the univariate logistic regression. Fit
with little/no regularization, since beta calibration is the maximum-likelihood (log-loss)
fit; strong L2 would shrink the feature weights `a, b` toward an intercept-only constant map.

## Working code

The implementation fits a logistic regression on the two transformed features and applies it
through the same featurization at predict time, mirroring the canonical `betacal` package.

```python
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression


class BetaCalibration:
    """Three-parameter beta calibration (Kull, Silva Filho & Flach).

    Fits mu_beta(s; a, b, c) = 1/(1 + 1/(e^c * s^a / (1-s)^b)) by bivariate
    logistic regression on the features (log s, -log(1-s)): the LR weights are
    (a, b) and the intercept is c."""

    def __init__(self, C=1e10):
        # near-zero regularization => the log-loss / MLE fit that defines the method.
        self.eps = 1e-6
        self.model_ = LogisticRegression(max_iter=2000, solver="lbfgs", C=C)
        self.map_ = None  # [a, b, m] for interpretation
        self.active_features_ = None

    def _featurize(self, probs):
        probs = np.asarray(probs).reshape(-1)
        p = np.clip(probs, self.eps, 1.0 - self.eps)        # keep log s, log(1-s) finite
        return np.column_stack([np.log(p), -np.log1p(-p)])  # (log s, -log(1-s))

    def fit(self, probs, labels, groups=None):
        X = self._featurize(probs)
        labels = np.asarray(labels).reshape(-1).astype(int)
        self.model_.fit(X, labels)                          # MLE of (a, b, c)
        coef = self.model_.coef_[0]

        # Canonical betacal monotonicity guard: if a or b is negative,
        # fix that coefficient to zero by dropping its feature and refitting.
        if coef[0] < 0:
            self.active_features_ = [1]
            self.model_.fit(X[:, self.active_features_], labels)
            a, b = 0.0, float(self.model_.coef_[0][0])
        elif coef[1] < 0:
            self.active_features_ = [0]
            self.model_.fit(X[:, self.active_features_], labels)
            a, b = float(self.model_.coef_[0][0]), 0.0
        else:
            self.active_features_ = [0, 1]
            a, b = map(float, coef)

        c = float(self.model_.intercept_[0])
        # midpoint m where LR(m) = 1, i.e. c = b ln(1-m) - a ln m (interpretation only)
        m = minimize_scalar(
            lambda mh: abs(b * np.log(1.0 - mh) - a * np.log(mh) - c),
            bounds=(self.eps, 1.0 - self.eps), method="bounded",
        ).x
        self.map_ = [float(a), float(b), float(m)]
        return self

    def predict_proba(self, probs, groups=None):
        X = self._featurize(probs)[:, self.active_features_]
        # 1/(1 + 1/exp(a*log s + b*(-log(1-s)) + c)) == mu_beta(s; a, b, c)
        return np.clip(self.model_.predict_proba(X)[:, 1], self.eps, 1.0 - self.eps)
```

Canonical sign convention note: the `betacal` reference package builds the features as
`[log s, log(1-s)]` and then negates the second column (`x[:,1] *= -1`), yielding
`(log s, -log(1-s))` so the learned coefficients are directly `(a, b)` with `a, b >= 0`; it
also applies the drop-and-refit monotonicity guard above. Building the second feature as
`+log(1-s)` instead (and reading its coefficient as `-b`) fits the identical map.

Symmetric variant beta[a=b] — univariate logistic regression on the log-odds:

```python
class BetaABCalibration:  # a = b
    def __init__(self, C=1e10):
        self.eps = 1e-6
        self.model_ = LogisticRegression(max_iter=2000, solver="lbfgs", C=C)

    def _featurize(self, probs):
        p = np.clip(np.asarray(probs).reshape(-1), self.eps, 1.0 - self.eps)
        return np.log(p / (1.0 - p)).reshape(-1, 1)         # log-odds feature

    def fit(self, probs, labels, groups=None):
        self.model_.fit(self._featurize(probs),
                        np.asarray(labels).reshape(-1).astype(int))
        return self

    def predict_proba(self, probs, groups=None):
        return np.clip(self.model_.predict_proba(self._featurize(probs))[:, 1],
                       self.eps, 1.0 - self.eps)
```
