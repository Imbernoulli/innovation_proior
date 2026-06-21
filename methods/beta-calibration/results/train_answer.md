A classifier score in [0,1] is rarely a probability. Calibration fixes this by learning a post-hoc map mu so that mu(f(x)) approximates the true positive-class posterior. The trouble is that the empirical map that makes the training set perfectly calibrated overfits wildly, so we need a parametric family with the right inductive bias. The standard choice is logistic or Platt calibration, which fits a sigmoid to the score. It is cheap and needs little data, but it comes from an equal-variance Gaussian assumption on a bounded score, and that assumption is too rigid: it puts mass outside [0,1], it can only spread scores toward the extremes and never pull overconfident extreme scores back in, and it does not even contain the identity map, so it can uncalibrate an already calibrated classifier. Isotonic calibration has the opposite problem: it can bend in any monotone direction, but with no parametric bias it overfits badly on the small held-out calibration sets that are common in practice. What is missing is a family that keeps the low cost and small-data friendliness of the sigmoid while being rich enough to represent both directions of distortion and to leave a calibrated model alone.

The method that closes this gap is beta calibration. Instead of modelling the per-class score distribution as a Gaussian, beta calibration uses the beta distribution, which is the natural density on [0,1]. Positing positives ~ Beta(alpha_1, beta_1) and negatives ~ Beta(alpha_0, beta_0) and forming the likelihood ratio gives LR(s) = e^c * s^a / (1-s)^b, where a = alpha_1 - alpha_0, b = beta_0 - beta_1, and K = B(alpha_1,beta_1)/B(alpha_0,beta_0) = e^{-c}. Under a uniform prior the calibrated posterior is mu_beta(s; a, b, c) = 1 / (1 + 1/(e^c * s^a / (1-s)^b)). This is a three-parameter family. It is monotone non-decreasing exactly when a, b >= 0, because the derivative of log LR is a/s + b/(1-s). It contains the sigmoid-like spread as the special case a = b > 1, the inverse-sigmoid gathering shape as a = b < 1, and the identity map as a = b = 1 with c = 0, so an already calibrated classifier is left untouched. Asymmetric distortions are reachable by taking a != b, which the sigmoid family cannot represent at all.

The practical payoff is that beta calibration is fitted by a single off-the-shelf logistic-regression call. Taking the log of the likelihood ratio yields ln LR(s) = a ln s - b ln(1-s) + c, which is linear in the two features ln s and -ln(1-s). Therefore mu_beta is exactly a bivariate logistic-regression posterior on those two transformed features with coefficients (a, b) and intercept c. Fitting by log-loss on the calibration set coincides with maximum likelihood, just as for Platt scaling, but now the family is far more flexible. The symmetric variant a = b collapses to univariate logistic regression on the log-odds ln(s/(1-s)), which retroactively explains the older linear-in-log-odds recalibration heuristic as a special case of beta calibration. A simple guard keeps the map monotone: fit unconstrained, and if either coefficient comes out negative, drop that feature, fix its coefficient to zero, and refit the remaining univariate logistic regression. The canonical fit uses essentially no regularization, because beta calibration is defined by the maximum-likelihood solution; heavy L2 shrinkage would pull the map toward an uninformative constant.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression


class BetaCalibration:
    """Beta calibration for binary classifier scores in (0, 1).

    Fits mu(s; a, b, c) = 1 / (1 + 1/(e^c * s^a / (1-s)^b)) by bivariate
    logistic regression on features (log s, -log(1-s)).
    """

    def __init__(self, C=1e10):
        self.eps = 1e-6
        self.model_ = LogisticRegression(
            solver="lbfgs", max_iter=2000, C=C
        )
        self.active_features_ = None

    def _featurize(self, probs):
        p = np.clip(np.asarray(probs).reshape(-1), self.eps, 1.0 - self.eps)
        return np.column_stack([np.log(p), -np.log1p(-p)])

    def fit(self, probs, labels):
        X = self._featurize(probs)
        y = np.asarray(labels).reshape(-1).astype(int)
        self.model_.fit(X, y)
        coef = self.model_.coef_[0]

        # Monotonicity guard: if a coefficient is negative, drop that feature.
        if coef[0] < 0:
            self.active_features_ = [1]
            self.model_.fit(X[:, self.active_features_], y)
        elif coef[1] < 0:
            self.active_features_ = [0]
            self.model_.fit(X[:, self.active_features_], y)
        else:
            self.active_features_ = [0, 1]
        return self

    def predict_proba(self, probs):
        X = self._featurize(probs)[:, self.active_features_]
        return np.clip(
            self.model_.predict_proba(X)[:, 1], self.eps, 1.0 - self.eps
        )
```
