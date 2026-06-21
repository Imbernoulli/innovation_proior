The S-learner confirmed the crossing I bet on, and the crossing is the whole map I need. On `ihdp_synth` the pooled single model won clearly — PEHE **0.800**, well below DR's **1.383** and the T-learner's **1.14** — because pooling the dominant $\exp(0.8x_0+0.5x_1)$ baseline once with all 747 rows and shrinking the simple effect is the variance-minimizing fit a small dataset rewards. On `acic_synth` the bet paid in the other direction: PEHE **0.524**, *worse* than DR's orthogonalized **0.501**, with ATE error **0.072** far worse than DR's **0.029** — the attenuation signature, a flag that loses the split competition against $p=50$ strong covariates producing a too-small *average* effect. So the three rungs have drawn the map cleanly: outcome-modeling-and-toggling wins where variance dominates (small, low-dimensional, simple-effect), orthogonalization wins where the surfaces are hard and the effect is real-but-weak against strong covariates, and they *cross*. The S-learner cannot win `acic_synth` because it has no orthogonalization; the DR-learner cannot win `ihdp_synth` because its per-unit AIPW correction is too high-variance. The method that beats both must do the contradictory-seeming thing: keep the orthogonalization but at *low variance* — residualize to get DR's confounding-robustness without building a noisy per-unit augmented score, and without the S-learner's attenuation.

I propose the *R-learner*, built on Robinson's residual-on-residual identity. Define the *marginal* outcome mean $m(x)=E[Y\mid X=x]$ — marginalizing over treatment, not the arm means — and compute $E[Y-m(X)\mid X,T]$. Under unconfoundedness $E[Y\mid X,T]=\mu_0(X)+T\,\tau(X)$ and $m(x)=\mu_0(x)+e(x)\tau(x)$, so subtracting,

$$Y-m(X)=(T-e(X))\,\tau(X)+\varepsilon,\qquad E[\varepsilon\mid X,T]=0.$$

This is Robinson's (1988) partially-linear form with one crucial change: Robinson's slope multiplying the regressor residual is a *constant* $\beta$; here the thing multiplying the treatment residual $T-e(X)$ is $\tau(X)$, a *function*. Promoting the slope to a function gives heterogeneous effects out of exactly the residual-on-residual structure proven robust to slow nuisance estimation — the same orthogonalization DR had, but expressed as a *loss* rather than a per-unit score. To estimate the function-valued slope, keep the squared-error form and let $\tau$ range over functions:

$$\tau(\cdot)=\arg\min_\tau E\!\left[\big((Y-m(X))-(T-e(X))\,\tau(X)\big)^2\right].$$

This really recovers $\tau$ and nothing else. Writing $Y-m=(T-e)\tau+\varepsilon$, then $(Y-m)-(T-e)\tau'=(T-e)(\tau-\tau')+\varepsilon$, and squaring with $E[\varepsilon\mid X,T]=0$ killing the cross term, the loss equals $E[(T-e)^2(\tau-\tau')^2]+E[\varepsilon^2]$. The second term is constant in $\tau'$; the first is a nonnegative overlap-weighted squared distance, since conditional on $X=x$ the binary $T$ gives $E[(T-e)^2\mid X=x]=e(x)(1-e(x))>0$ under overlap. So the minimizer is uniquely $\tau$. CATE estimation has become *plain regularized squared-loss minimization in $\tau$*, with the confounding control living entirely in the loss through the residualization and the learner free to express the heterogeneity — exactly the separation I wanted, and no high-variance per-unit score.

I do not know $m$ and $e$, so the feasible method cross-fits them — $\hat m(x)=E[Y\mid X]$ by a regressor, $\hat e(x)=P(T=1\mid X)$ by a classifier, predicted out-of-fold so the residuals $\tilde Y=Y-\hat m$ and $\tilde T=T-\hat e$ are out-of-sample. Plugging in estimates does not propagate error the way the T-learner's did: expanding the feasible loss against the oracle, the nuisance error enters through a *product* of the two nuisance errors plus three single-error cross terms, and cross-fitting centers each single-error term to mean zero through $E[Y-m\mid X]=0$ and $E[T-e\mid X]=0$. The leading penalty is the product $\|\hat m-m\|\cdot\|\hat e-e\|$, second order, so $\hat\tau$ inherits the *oracle* rate set by $\tau$'s complexity — the quasi-oracle property. This is the same product-of-errors robustness DR had, reached through a loss rather than a score, and the difference I am counting on is *variance*: DR's pseudo-outcome carried three estimated surfaces and two inverse-propensity terms per unit, while the R-loss carries only $\tilde Y$ and $\tilde T$ — two residuals, no explicit $1/\hat e$ multiplying a raw outcome.

To make the objective something a standard learner accepts, factor the integrand: $[\tilde Y-\tilde T\,\tau(X)]^2=\tilde T^2[\tilde Y/\tilde T-\tau(X)]^2$. So the R-loss is *identically* a weighted least-squares regression — regress the pseudo-outcome $\tilde Y/\tilde T$ on $X$ with sample weight $\tilde T^2$. Every learner I care about accepts sample weights, so the whole thing is a single weighted-regression call, and the weight does exactly the right work: it is small precisely where $\tilde T$ is small — where treatment was nearly deterministic given $X$, so the pseudo-outcome $\tilde Y/\tilde T$ is explosive and carries almost no information about $\tau$. The $1/\tilde T$ blow-up is *cancelled* by the $\tilde T^2$ weight, leaving only the structurally informative reweighting.

But "cancelled in the loss" is not "cancelled in the implementation," and this is where the fill is deliberately more defensive than the clean identity. A learner minimizing weighted squared error still *forms* the target $\tilde Y/\tilde T$ numerically, and when $\tilde T$ is within floating-point distance of zero that target is a huge number times a tiny weight — a $0\times\infty$ that, with finite precision and a finite-depth tree, does not perfectly cancel and can still tug a leaf; on `jobs_synth`, where $\tilde Y$ is earnings-scale, a near-zero $\tilde T$ produces a pseudo-outcome of astronomical magnitude. So the fill floors the divisor, `safe_T = where(|T_res| > 0.01, T_res, sign(T_res)*0.01 + 1e-8)`, capping $|\tilde T|$ at $0.01$ so the formed pseudo-outcome can never exceed $\sim 100\,\tilde Y$ — a finite-sample numerical guard, not a change to the population loss, touching only the handful of near-deterministic-treatment units the $\tilde T^2$ weight was already suppressing. It then clips the pseudo-outcome itself, `q = percentile(|pseudo|, 95); pseudo = clip(pseudo, -q, q)`, the same tail control DR's 97th-percentile winsorization applied, but a *narrower* tail because a ratio $\tilde Y/\tilde T$ has heavier numerical tails than DR's additive score. Neither guard is in the textbook R-loss; both are the price of forming the ratio target numerically, and skipping them is how the R-learner earns its reputation for instability on weak-overlap, earnings-scale data exactly like `jobs_synth`.

For the scaffold edit: cross-fit $\hat m$ (a `GradientBoostingRegressor`, depth 4) and $\hat e$ (a `GradientBoostingClassifier`, depth 3) with `KFold(5)`, predicting out-of-fold; clip $\hat e$ to $[0.05,0.95]$ (the same overlap floor every rung used, here bounding $|\tilde T|$ below for binary $T$); form $\tilde Y$ and $\tilde T$; build the weight $\tilde T^2$ and the guarded, clipped pseudo-outcome; and fit a *shallower, slower* `GradientBoostingRegressor` (depth 3, learning rate 0.05) on $X\to\text{pseudo}$ with `sample_weight=weights`. The final learner is deliberately smoother than the nuisance learners because it estimates $\tau$, the simplest function in play — the same bet DR made, now without DR's augmentation variance. Seeds split `seed`, `seed+1`, `seed+2`.

This is the rung I expect to finally beat the surface methods *across* the board rather than complementarily. On `acic_synth`, where the S-learner attenuated (PEHE **0.524**, ATE error **0.072**), R's orthogonalization should fix the attenuation — PEHE *at or below* DR's **0.501** and ATE error *far below* the S-learner's **0.072**. On `ihdp_synth`, where the S-learner won with **0.800** and DR lost with **1.383**, the test is whether R's lower-variance orthogonalization can *hold near the S-learner's level* rather than paying DR's small-sample tax: I expect PEHE close to **0.800**, decisively better than DR, because two residuals carry less variance than DR's five-object score, so the orthogonalization survives $n=747$. On `jobs_synth` I expect R to beat the S-learner's **396** PEHE *and* **51.0** ATE error, with the divisor guard and 95th-percentile clip preventing any ratio-target blow-up. The sharpest test of the whole thesis: if "orthogonalization at low variance" is right, R should be the first rung *not* beaten by a simpler method on any dataset, with the best ATE errors overall, because it is the only construction with both the residualization (beating the S-learner's attenuation) and the low variance (beating DR's augmentation noise). If it is held back anywhere, it should be on `ihdp_synth`, where even two residuals must be cross-fit on 747 rows — that residual-estimation error on the smallest dataset, and the single-regressor final stage smoothing a heavy-tailed ratio target, is the one weakness I would hand to a stronger forest-based successor.

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
