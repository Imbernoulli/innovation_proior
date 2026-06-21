The T-learner's headline prediction held: dropping the $1/\hat e$ division crushed IPW's `jobs_synth` catastrophe, PEHE falling from **5183.8** to **655.3** and ATE error from **843.3** to **30.2**, because $\hat\tau$ is now a difference of bounded regression predictions instead of a sum of inflated weighted outcomes. But the T-learner was *not* the best surface method even at this rung — `ihdp_synth` PEHE **1.14** and, tellingly, `acic_synth` PEHE **0.77**, both noticeably worse than I want. The `acic_synth` weakness is exactly what I flagged: with $p=50$, a complex shared baseline, and locally thin treated regions, fitting two hard 50-dimensional surfaces *in isolation* and subtracting them pays the full response-surface rate twice and corrupts the (simpler) difference with two uncorrelated errors. So the diagnosis is precise — the surfaces are estimated in isolation, at the response rate, and $\hat\tau$ is *first-order* sensitive to their error. I want the estimator's error to be *second-order* in nuisance error.

What "second-order" should mean is best fixed against an oracle. If I could *see* the individual contrast $Y(1)-Y(0)$ for each unit, I would regress it on $X$ and get an estimate whose rate is set by $\tau$'s own smoothness alone. The T-learner does not come close — it pays the surface rate. IPW manufactured an observable with conditional mean $\tau(x)$, recovering that adaptivity, but it was singly robust and exploded. So the real question is whether I can manufacture a per-unit pseudo-outcome, observable from $(X,T,Y)$, whose conditional mean is $\tau(x)$ *and* whose bias under estimated nuisances is a product of errors rather than a single first-order term.

I propose the *DR-learner* (doubly-robust / AIPW), built from the augmented-IPW score. For the treated mean $E[Y(1)]=E[\mu_1(X)]$ the efficient influence function is $\phi=\mathbb 1(T=1)/e(X)\cdot\{Y-\mu_1(X)\}+\mu_1(X)$ — a regression-imputation piece $\mu_1$ augmented by an inverse-probability-weighted *residual* that corrects the imputation only on treated units. Its von Mises expansion makes the bias of the ATE estimator an integral of a *product* $\int\{1/\bar e-1/e\}\{\mu_1-\bar\mu_1\}\,e\,dx$ — propensity error times outcome error, so two slow $n^{-1/3}$ rates multiply to a smaller-order $n^{-2/3}$ — and it is doubly robust: get $\bar e=e$ and the first factor vanishes, get $\bar\mu_1=\mu_1$ and the second vanishes. That is precisely the mutual insurance IPW (which threw away the outcome model) and the T-learner (which threw away the propensity) each lacked. To estimate the *conditional* effect I regress the conditional analogue on $X$ rather than averaging it. The two-arm uncentered influence function is

$$\phi=\mu_1(X)-\mu_0(X)+\frac{T\,(Y-\mu_1(X))}{e(X)}-\frac{(1-T)\,(Y-\mu_0(X))}{1-e(X)},$$

a regression-difference $\mu_1-\mu_0$ plus an IPW correction applied to *each arm's residual*. This unifies the two prior rungs: drop the augmentation and keep only $\mu_1-\mu_0$ and I am back at the T-learner (first-order, no double robustness); drop the outcome regression and set $\mu_w\equiv 0$ and the pseudo-outcome becomes the IPW transform (singly robust, high variance). The DR pseudo-outcome keeps *both* pieces, so the $\mu_w$ augmentation subtracts off the predictable part of $Y$ *before* the residual is weighted — the very thing that would have tamed IPW's `jobs_synth` spikes, since the weight now multiplies a residual $Y-\hat\mu$ rather than a raw earnings-scale $Y$.

The oracle check confirms the target. With true nuisances, condition on $X=x$: the $\mu_1-\mu_0$ term passes through; when $T=1$ (probability $e$) the correction contributes $e\cdot E[(Y-\mu_1)/e\mid X,T=1]=\mu_1-\mu_1=0$, and the $T=0$ term vanishes the same way, so $E[\phi\mid X]=\mu_1-\mu_0=\tau$. But I estimate $\hat e,\hat\mu_0,\hat\mu_1$, form $\hat\phi$, and regress that, so the real object is the conditional bias $\hat b(x)=E[\hat\phi-\phi\mid\text{nuisances},X=x]$. Grinding it out arm by arm — the $T=1$ branch contributes $e(\mu_1-\hat\mu_1)/\hat e$ from the weighted residual and $\hat\mu_1-\mu_1$ from the imputation, collecting to $(\hat e-e)(\hat\mu_1-\mu_1)/\hat e$, the $T=0$ branch symmetrically — gives

$$\hat b(x)=\frac{\hat e-e}{\hat e}\,(\hat\mu_1-\mu_1)+\frac{\hat e-e}{1-\hat e}\,(\hat\mu_0-\mu_0).$$

There is the function-valued echo of the scalar product: the conditional bias is a *product* of propensity error and outcome error, summed over arms, with the propensity entering only through a bounded reweighting. Double robustness is manifest — $\hat e=e$ or $\hat\mu_w=\mu_w$ kills every term — so the second-stage error relative to the oracle can be of *smaller order* than either nuisance error, meaning $\hat\tau$ converges at the complexity of $\tau$ even when the nuisances are estimated slowly. That is exactly what the T-learner's **0.77** on `acic_synth` was missing: there both surfaces are slow, but if their errors enter $\hat\tau$ only as a product with the propensity error, the effect estimate need not inherit their slowness.

The derivation has one cheat that, left in, would silently reintroduce a first-order leak: I computed $\hat b$ treating the nuisances as *fixed*. If $\hat\mu_1$ is fit on the same rows I then evaluate $\hat\phi$ on, the residual $Y-\hat\mu_1$ is correlated with $\hat\mu_1$ because $Y$ helped fit it, $E[(Y-\hat\mu_1)\mid T=1]$ is no longer $\mu_1-\hat\mu_1$, and the error carries an empirical-process term that does not vanish without Donsker restrictions gradient boosting violates. The cure is *cross-fitting*: estimate the nuisances on one part of the data and evaluate the pseudo-outcome on a disjoint part, so conditional on the training part the nuisances are genuinely fixed and independent of the evaluation units, the clean $\hat b$ holds, and the contamination term has conditional mean zero. With $K=5$ folds I train out-of-fold and predict in-fold, so every unit's pseudo-outcome uses nuisances that never saw it; $K=5$ keeps each training fold large while keeping the out-of-fold predictions honest. This is not a technicality — it is what makes the product-bias derivation true rather than aspirational, and it is the structural piece the T-learner never needed because it never had a residual to corrupt.

The implementation cross-fits with `KFold(5)`: per fold it fits $\hat\mu_0$ on the control arm of the training portion, $\hat\mu_1$ on the treated arm, and $\hat e$ on the full training portion, predicting all three out-of-fold, with a guard (`mask.sum() > 5`) falling back to the arm mean when a fold's arm is too thin — a small-sample safety that matters on `ihdp_synth`'s $n=747$ split five ways. It clips $\hat e$ to $[0.05,0.95]$ so the $1/\hat e$ and $1/(1-\hat e)$ in the correction stay bounded — the same overlap floor IPW used, now guarding a *residual*-weighting rather than a raw-outcome weighting, so far more effective. Then it forms the AIPW pseudo-outcome in closed form and *winsorizes* it at the 97th percentile of $|\hat\phi|$: even with clipped propensities, heavy-tailed outcome residuals near weak overlap can produce a few extreme pseudo-outcomes, and capping them without touching the bulk is cheap insurance — the tail control IPW lacked. Finally a *shallower, more strongly regularized* `GradientBoostingRegressor` (depth 3, learning rate 0.05) regresses $X\to\hat\phi$, deliberately smoother than the depth-4 nuisance learners because it is estimating $\tau$, which I bet is simpler than the surfaces — a smoother final fit is the whole point of regressing a pseudo-outcome instead of differencing two complex surfaces. Seeds split `seed`, `seed+1`, `seed+2`.

The cleanest test is `acic_synth`, where the T-learner was weakest (**0.77**): the product-of-errors bias should let DR converge nearer $\tau$'s complexity, so I expect its PEHE *at or below* **0.77** and in the same ballpark as the other orthogonalized methods — the rung's reason to exist. On `ihdp_synth` ($n=747$, the smallest sample) the prediction is guarded: cross-fitting splits an already-small dataset and the $1/\hat e$ residual correction adds variance the winsorization only partly tames, so DR need *not* beat the T-learner's **1.14** there — if it lands around or slightly above, that is the expected small-sample tax, not a contradiction. On `jobs_synth` I expect DR in the same hundreds regime as the T-learner's **655.3**, since the augmentation weights residuals not raw earnings, with the ATE error staying low because the doubly-robust construction estimates the average cleanly. That residual-correction variance — useful asymptotically, expensive in small samples — is the diagnosis I hand to the next rung, which will ask whether I can get the orthogonalization's confounding-robustness *without* paying the full augmentation variance, by residualizing once and fitting the effect directly rather than building a per-unit AIPW score.

```python
class CATEEstimator(BaseCATEEstimator):
    """DR-Learner: Doubly Robust CATE estimation.

    Steps:
    1. Cross-fit nuisance models:
       - mu0(X) = E[Y|X, T=0], mu1(X) = E[Y|X, T=1]  (outcome models)
       - e(X) = P(T=1|X)  (propensity score)
    2. Compute doubly-robust pseudo-outcomes:
       phi(X) = mu1(X) - mu0(X)
              + T*(Y - mu1(X))/e(X)
              - (1-T)*(Y - mu0(X))/(1-e(X))
    3. Fit a final CATE model on X -> phi(X)
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

        # Cross-fit nuisance models
        kf = KFold(n_splits=5, shuffle=True, random_state=self._seed)
        mu0_hat = np.zeros(n)
        mu1_hat = np.zeros(n)
        e_hat = np.zeros(n)

        for train_idx, val_idx in kf.split(X):
            # Outcome models (separate for T=0 and T=1)
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

            # Propensity model
            mt = self._make_model_t()
            mt.fit(X[train_idx], T[train_idx])
            e_hat[val_idx] = mt.predict_proba(X[val_idx])[:, 1]

        # Clip propensity scores
        e_hat = np.clip(e_hat, 0.05, 0.95)

        # Doubly-robust pseudo-outcomes
        pseudo = (
            mu1_hat - mu0_hat
            + T * (Y - mu1_hat) / e_hat
            - (1 - T) * (Y - mu0_hat) / (1 - e_hat)
        )

        # Clip extreme pseudo-outcomes
        q = np.percentile(np.abs(pseudo), 97)
        pseudo = np.clip(pseudo, -q, q)

        # Fit CATE model on pseudo-outcomes
        self._cate_model = self._make_cate_model()
        self._cate_model.fit(X, pseudo)
        return self

    def predict(self, X):
        return self._cate_model.predict(X)
```
