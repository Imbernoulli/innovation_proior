**Problem (from step 4).** The S-learner and DR-learner *cross*: pooling-and-toggling wins where
variance dominates (`ihdp_synth` 0.800) but attenuates the real effect against strong covariates
(`acic_synth` 0.524, ATE error 0.072 — worse than DR's 0.501/0.029); DR orthogonalizes but its per-unit
score is too high-variance for the small sets. The method that beats both must keep orthogonalization
*at low variance*.

**Key idea.** Robinson residualize: $Y-m(X)=(T-e(X))\tau(X)+\varepsilon$ promotes the constant slope to a
*function* $\tau(X)$. The population least-squares projection recovers $\tau$, and factoring
$[\tilde Y-\tilde T\tau]^2=\tilde T^2[\tilde Y/\tilde T-\tau]^2$ makes it one *weighted regression* of
$\tilde Y/\tilde T$ on $X$ with weight $\tilde T^2$. Confounding control lives in the loss; the learner
just expresses heterogeneity.

**Why it works.** Same product-of-errors (quasi-oracle) robustness as DR, but through a loss built from
only *two* residuals — no $1/\hat e$ times a raw outcome, no three-surface score — so far lower variance.
The $\tilde T^2$ weight cancels the $1/\tilde T$ pseudo-outcome blow-up and downweights
near-deterministic-treatment units.

**Hyperparameters.** Cross-fit (`KFold(5)`) $\hat m$ (`GradientBoostingRegressor`, depth 4) and $\hat e$
(`GradientBoostingClassifier`, depth 3); clip $\hat e\in[0.05,0.95]$. Final learner shallower/slower
(depth 3, lr 0.05), weighted by $\tilde T^2$. **Numerical guards (beyond the textbook R-loss):** floor
$|\tilde T|$ at 0.01 (`safe_T`) so the formed ratio target can't explode in finite precision, and clip
the pseudo-outcome at its 95th percentile (a narrower tail than DR's 97th — a ratio has heavier tails).

**What to watch.** R should be the first rung not beaten by a simpler method anywhere: fix the
S-learner's `acic_synth` attenuation (PEHE/ATE at or below DR), hold near the S-learner on `ihdp_synth`
(two residuals survive $n=747$ where DR's score didn't), and beat the S-learner on `jobs_synth` with the
guards preventing a ratio blow-up. Residual-estimation error on the smallest set is its one weakness.

```python
class CATEEstimator(BaseCATEEstimator):
    """R-Learner: Robinson decomposition for CATE estimation.

    Based on the Robinson (1988) decomposition:
        Y - m(X) = (T - e(X)) * tau(X) + epsilon

    Steps:
    1. Cross-fit nuisance models:
       - m(X) = E[Y|X]  (marginal outcome model)
       - e(X) = P(T=1|X)  (propensity score)
    2. Compute residuals: Y_tilde = Y - m(X), T_tilde = T - e(X)
    3. Estimate tau(X) by minimizing: sum_i (Y_tilde_i - T_tilde_i * tau(X_i))^2
       This is equivalent to weighted least squares with weight T_tilde^2.
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

    def fit(self, X, T, Y):
        n = len(Y)

        # Cross-fit nuisance models
        kf = KFold(n_splits=5, shuffle=True, random_state=self._seed)
        m_hat = np.zeros(n)
        e_hat = np.zeros(n)

        for train_idx, val_idx in kf.split(X):
            # Outcome model E[Y|X]
            my = self._make_model_y()
            my.fit(X[train_idx], Y[train_idx])
            m_hat[val_idx] = my.predict(X[val_idx])

            # Propensity model P(T=1|X)
            mt = self._make_model_t()
            mt.fit(X[train_idx], T[train_idx])
            e_hat[val_idx] = mt.predict_proba(X[val_idx])[:, 1]

        # Clip propensity scores
        e_hat = np.clip(e_hat, 0.05, 0.95)

        # Residuals
        Y_tilde = Y - m_hat
        T_tilde = T - e_hat

        # R-Learner: pseudo-outcome = Y_tilde / T_tilde
        # Weight = T_tilde^2 (higher weight where treatment variation is larger)
        weights = T_tilde ** 2
        # Avoid division by zero
        safe_T = np.where(np.abs(T_tilde) > 0.01, T_tilde, np.sign(T_tilde) * 0.01 + 1e-8)
        pseudo = Y_tilde / safe_T

        # Clip extreme pseudo-outcomes
        q = np.percentile(np.abs(pseudo), 95)
        pseudo = np.clip(pseudo, -q, q)

        # Weighted regression for CATE
        # Use sample_weight = T_tilde^2 to prioritize informative samples
        self._cate_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=self._seed + 2,
        )
        self._cate_model.fit(X, pseudo, sample_weight=weights)
        return self

    def predict(self, X):
        return self._cate_model.predict(X)
```
