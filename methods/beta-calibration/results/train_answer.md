A classifier score in [0,1] is rarely a probability. Calibration fixes this by learning a post-hoc map mu so that mu(f(x)) approximates the true positive-class posterior. The trouble is that the empirical map that makes the training set perfectly calibrated overfits wildly, so we need a parametric family with the right inductive bias. The standard choice is logistic or Platt calibration, which fits a sigmoid to the score. It is cheap and needs little data, but it comes from an equal-variance Gaussian assumption on a bounded score, and that assumption is too rigid: it puts mass outside [0,1], it can only spread scores toward the extremes and never pull overconfident extreme scores back in, and it does not even contain the identity map, so it can uncalibrate an already calibrated classifier. Isotonic calibration has the opposite problem: it can bend in any monotone direction, but with no parametric bias it overfits badly on the small held-out calibration sets that are common in practice. What is missing is a family that keeps the low cost and small-data friendliness of the sigmoid while being rich enough to represent both directions of distortion and to leave a calibrated model alone.

The method that closes this gap is beta calibration. Instead of modelling the per-class score distribution as a Gaussian, beta calibration uses the beta distribution, which is the natural density on [0,1]. Positing positives ~ Beta(alpha_1, beta_1) and negatives ~ Beta(alpha_0, beta_0) and forming the likelihood ratio gives LR(s) = e^c * s^a / (1-s)^b, where a = alpha_1 - alpha_0, b = beta_0 - beta_1, and K = B(alpha_1,beta_1)/B(alpha_0,beta_0) = e^{-c}. Under a uniform prior the calibrated posterior is mu_beta(s; a, b, c) = 1 / (1 + 1/(e^c * s^a / (1-s)^b)). This is a three-parameter family. It is monotone non-decreasing exactly when a, b >= 0, because the derivative of log LR is a/s + b/(1-s). It contains the sigmoid-like spread as the special case a = b > 1, the inverse-sigmoid gathering shape as a = b < 1, and the identity map as a = b = 1 with c = 0, so an already calibrated classifier is left untouched. Asymmetric distortions are reachable by taking a != b, which the sigmoid family cannot represent at all.

The practical payoff is that beta calibration is fitted by a single off-the-shelf logistic-regression call. Taking the log of the likelihood ratio yields ln LR(s) = a ln s - b ln(1-s) + c, which is linear in the two features ln s and -ln(1-s). Therefore mu_beta is exactly a bivariate logistic-regression posterior on those two transformed features with coefficients (a, b) and intercept c. Fitting by log-loss on the calibration set coincides with maximum likelihood, just as for Platt scaling, but now the family is far more flexible. The symmetric variant a = b collapses to univariate logistic regression on the log-odds ln(s/(1-s)), which retroactively explains the older linear-in-log-odds recalibration heuristic as a special case of beta calibration. A simple guard keeps the map monotone: fit unconstrained, and if either coefficient comes out negative, drop that feature, fix its coefficient to zero, and refit the remaining univariate logistic regression. The canonical fit uses essentially no regularization, because beta calibration is defined by the maximum-likelihood solution; heavy L2 shrinkage would pull the map toward an uninformative constant. After fitting, it is convenient to recover the interpretable midpoint m where mu_beta = 1/2, i.e. where LR(m) = 1: since c = b ln(1-m) - a ln m, solving that one-dimensional equation for m gives a location parameter directly comparable to the sigmoid's own midpoint, even though, unlike the sigmoid, moving m here reshapes the curve rather than just sliding it. Both `fit` and `predict_proba` also carry a `groups` argument that this implementation ignores, kept only for interface compatibility with group-wise variants of the calibrator; this binary form is group-agnostic.

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
