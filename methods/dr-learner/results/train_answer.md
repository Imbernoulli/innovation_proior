We have observational data consisting of covariates X, a binary treatment A, and an outcome Y, and we want to estimate the conditional average treatment effect tau(x) = E(Y^1 - Y^0 | X = x), the whole heterogeneous function of who benefits and by how much. Under the standard causal assumptions of no unmeasured confounding, consistency, and positivity, this contrast is identified from observed data, but estimating it well is hard for three interacting reasons: treatment assignment depends on covariates, the effect can vary nonlinearly across X, and the individual response surfaces mu_a(x) = E(Y | X=x, A=a) can be much more complex than their difference tau. The central empirical fact is that tau may be nearly constant or very smooth even when mu_0 and mu_1 are awkward, wiggly functions, so any estimator whose error is driven by the surfaces themselves will do unnecessary work and can produce spurious heterogeneity.

Existing approaches each leave a gap. The T-learner fits mu_1 on the treated and mu_0 on the controls separately and differences them; this is the natural plug-in, but because the two arms are fit on largely disjoint, confounding-skewed subsamples, the difference inherits the larger of the two surface errors and cannot exploit a simple tau. Inverse-probability-weighting transforms (A - pi(X)) Y / {pi(X)(1 - pi(X))} into a pseudo-outcome with conditional mean tau, which does adapt to tau's complexity, but it throws away the outcome models and divides by pi(1-pi), so it explodes wherever overlap is weak and is only consistent if the propensity is exactly right. The S-learner pools all data into one model mu(x,a) and toggles a, which is lower variance but lets regularization shrink the treatment feature and attenuate real heterogeneity. The R-learner residualizes both Y and A and fits the effect through a weighted ratio, which is orthogonal but can still be noisy in small samples because of the 1/(A - pi) pseudo-target.

The method I propose is the DR-Learner, short for Doubly Robust Learner. It is the function-valued analog of the augmented inverse-probability-weighted estimator for the average treatment effect: instead of averaging the efficient influence function, we regress it on X. Stage one cross-fits three flexible nuisance models, the propensity pi(x) = P(A=1 | X=x) and the two arm-specific outcome regressions mu_0(x) and mu_1(x). From these we build the per-unit doubly robust pseudo-outcome

phi_hat = mu_1_hat(X) - mu_0_hat(X) + A (Y - mu_1_hat(X)) / pi_hat(X) - (1 - A) (Y - mu_0_hat(X)) / (1 - pi_hat(X)).

Stage two is simply a regression of phi_hat on X. The final model therefore targets tau directly rather than differencing two surface estimates.

The DR-Learner combines the strengths of the previous rungs while removing their weaknesses. With true nuisances, the correction term has conditional mean zero, so E(phi | X=x) = tau(x); regressing phi on X would recover tau at tau's own smoothness or sparsity rate, just as if we could observe the individual contrasts Y^1 - Y^0. With estimated nuisances, the conditional bias of phi_hat factors into a product of the propensity error and the outcome error on each arm: if either pi_hat = pi or mu_a_hat = mu_a, the bias vanishes entirely. That product structure means the second-stage error can be much smaller than the individual nuisance errors, so the final CATE estimate can converge at the complexity of tau even when the nuisances are estimated slowly. Cross-fitting is essential here: by training the nuisance models out-of-fold and predicting in-fold, we make each unit's pseudo-outcome independent of the models that produced it, which is what keeps the product bias valid for flexible machine learners without Donsker-style complexity restrictions.

In practice I use gradient boosted trees for the nuisances, a classifier for the propensity and regressors for the two arms, with the final CATE regressor set shallower and slower because tau is expected to be the simplest function in play. I clip estimated propensities into [0.05, 0.95] to enforce overlap and keep inverse weights bounded, and I winsorize extreme pseudo-outcomes at the 97th percentile of their absolute values so a few heavy residual tails do not distort the second-stage fit. Thin-arm folds fall back to the arm mean when there are too few treated or control units to fit a model.

```python
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


class CATEEstimator:
    """DR-Learner: cross-fitted doubly-robust pseudo-outcome regressed on X.

    Stage 1 fits the propensity and the two arm-specific outcome models out-of-fold.
    Stage 2 forms the augmented-IPW pseudo-outcome and regresses it on X.
    """

    def __init__(self):
        self._seed = int(os.environ.get("SEED", "42"))

    def _make_model_y(self):
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=self._seed,
        )

    def _make_model_t(self):
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=self._seed + 1,
        )

    def _make_cate_model(self):
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=self._seed + 2,
        )

    def fit(self, X, T, Y):
        n = len(Y)

        # Cross-fit nuisances: train out-of-fold, predict in-fold.
        kf = KFold(n_splits=5, shuffle=True, random_state=self._seed)
        mu0_hat = np.zeros(n)
        mu1_hat = np.zeros(n)
        e_hat = np.zeros(n)

        for train_idx, val_idx in kf.split(X):
            mask0_train = T[train_idx] == 0
            mask1_train = T[train_idx] == 1

            m0 = self._make_model_y()
            m1 = self._make_model_y()

            if mask0_train.sum() > 5:
                m0.fit(X[train_idx[mask0_train]], Y[train_idx[mask0_train]])
                mu0_hat[val_idx] = m0.predict(X[val_idx])
            else:
                mu0_hat[val_idx] = Y[T == 0].mean() if (T == 0).sum() > 0 else Y.mean()

            if mask1_train.sum() > 5:
                m1.fit(X[train_idx[mask1_train]], Y[train_idx[mask1_train]])
                mu1_hat[val_idx] = m1.predict(X[val_idx])
            else:
                mu1_hat[val_idx] = Y[T == 1].mean() if (T == 1).sum() > 0 else Y.mean()

            mt = self._make_model_t()
            mt.fit(X[train_idx], T[train_idx])
            e_hat[val_idx] = mt.predict_proba(X[val_idx])[:, 1]

        # Enforce overlap so inverse weights stay bounded.
        e_hat = np.clip(e_hat, 0.05, 0.95)

        # Doubly-robust pseudo-outcome.
        pseudo = (
            mu1_hat - mu0_hat
            + T * (Y - mu1_hat) / e_hat
            - (1 - T) * (Y - mu0_hat) / (1 - e_hat)
        )

        # Winsorize extreme pseudo-outcomes.
        q = np.percentile(np.abs(pseudo), 97)
        pseudo = np.clip(pseudo, -q, q)

        # Stage 2: regress X -> pseudo-outcome to recover tau(x).
        self._cate_model = self._make_cate_model()
        self._cate_model.fit(X, pseudo)
        return self

    def predict(self, X):
        return self._cate_model.predict(X)
```
