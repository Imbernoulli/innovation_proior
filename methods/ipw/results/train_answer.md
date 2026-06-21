We have observational data on $N$ units — pretreatment covariates $X_i$, a binary treatment indicator $T_i \in \{0,1\}$, and a scalar outcome $Y_i$ — and we want the causal effect of the treatment on the outcome. In the potential-outcome framing each unit carries two latent outcomes $Y_i(1)$ and $Y_i(0)$, of which we observe only the one matching the treatment actually received, $Y_i = T_i Y_i(1) + (1-T_i) Y_i(0)$. The targets are the average treatment effect $\tau = E[Y(1) - Y(0)]$ and, more ambitiously, the conditional (heterogeneous) effect $\tau(x) = E[Y(1) - Y(0) \mid X = x]$. The structural difficulty is that we never see both potential outcomes for the same unit, so the effect contrasts an observed quantity against a counterfactual one. The reflex fix — compare the average treated outcome to the average control outcome, $E[Y \mid T=1] - E[Y \mid T=0]$ — is biased whenever treatment was not randomized: assignment happened out in the world by a process we do not control, the treated and untreated differ systematically in $X$, and part of the outcome gap is that $X$ difference rather than the treatment. That is confounding, and any solution must disentangle the effect of the treatment from the effect of being the kind of unit that tends to get treated.

To make anything recoverable we grant two assumptions. Unconfoundedness, $(Y(0), Y(1)) \perp T \mid X$, says that within a cell of units sharing the same $X$, who got treated is as-good-as-random. Overlap, $0 < e(x) < 1$ for every $x$, says every kind of unit had a genuine chance of either treatment, so there is no region of covariate space with only one arm to contrast. Under these the conditional response surfaces are identified, $E[Y(w) \mid X=x] = E[Y \mid T=w, X=x] =: \mu_w(x)$, and so $\tau(x) = \mu_1(x) - \mu_0(x)$ with $\tau = E_X[\mu_1(X) - \mu_0(X)]$. The obvious route from here is regression: fit $\mu_1$ and $\mu_0$ — either one pooled model with $T$ as a feature, or two separate per-arm surfaces — and difference them. But this dumps the entire burden onto getting one or two high-dimensional, nonlinear functions $\mu_w(x)$ exactly right, which is precisely the regime where flexible learners are least trustworthy. Two independently fit, separately regularized surfaces can carry systematically different biases whose difference is spurious heterogeneity; the single-model version, when $T$ is one weak feature among many strong covariates, barely splits on it and shrinks the effect toward zero. Worse, the regression route never uses the structure the assignment side hands us for free, and it has no second line of defense when the outcome model is wrong. Subclassification and matching on the propensity score do use the assignment side, but only coarsely — binning leaves residual within-stratum imbalance and arbitrary edges, matching discards units and depends on caliper and match count, and both return a number rather than a smooth function $\tau(x)$.

I propose Inverse Probability Weighting (IPW). The object that captures exactly the contaminating dependence of treatment on $X$ is the propensity score $e(x) = \Pr(T=1 \mid X=x) = E[T \mid X=x]$ — one number per unit, however high-dimensional $X$ is. It is a balancing score: $\Pr(T=1 \mid X, e(X)) = \Pr(T=1\mid X) = e(X)$ depends on $X$ only through $e(X)$, so $X \perp T \mid e(X)$, and it is the coarsest such score (any balancing score $b(x)$ satisfies $e(x) = f(b(x))$). Unconfoundedness given the full $X$ therefore implies unconfoundedness given the scalar $e(X)$: the confounding lives in a one-dimensional object. The mechanical picture is that the treated group is a non-representative sample of the population, drawn with inclusion probability $e(x)$, so high-$e$ units are over-represented among the treated and under-represented among the controls. That is a problem survey sampling solved cleanly. Horvitz and Thompson asked for an unbiased linear estimator of a population total $S = \sum_i X_i$ from a sample where element $u_i$ is included with probability $P(u_i)$; for $\sum_{i \in \text{sample}} \beta_i x_i$ to have expectation $\sum_i P(u_i)\beta_i X_i$ equal to $S$ for every possible population vector, we need $P(u_i)\beta_i = 1$ term by term, which forces $\beta_i = 1/P(u_i)$ with no freedom. Inverse-inclusion-probability weighting is not a heuristic; it is the unique unbiased choice.

Carrying that over, weight each treated outcome by $1/e(X)$. The cancellation is the whole trick. Conditioning on $X$ and using unconfoundedness, $E[TY/e(X) \mid X] = (1/e(X))\,E[T \mid X]\,E[Y(1) \mid X] = (1/e(X))\,e(X)\,\mu_1(X) = \mu_1(X)$, so averaging over $X$ gives $E[TY/e(X)] = E[Y(1)]$; symmetrically a control unit is "included" with probability $1-e(X)$, and $E[(1-T)Y/(1-e(X))] = E[Y(0)]$. The over-representation of high-$e$ units is exactly undone by dividing by $e(X)$. Hence
$$\tau = E\!\left[\frac{T\,Y}{e(X)} - \frac{(1-T)\,Y}{1-e(X)}\right],$$
an expectation of a single per-unit quantity involving only the outcome and the propensity, with no outcome model at all. Since $e(x)$ is unknown in observational data, I estimate it with a probability classifier $\hat e(x)$ — a logit when assignment is roughly linear, a gradient-boosted classifier when it is nonlinear with interactions — using $\mathtt{predict\_proba}$ for the probability rather than a hard label, and plug $\hat e$ into the weighting.

Two instabilities then have to be handled, both visible in the Horvitz–Thompson variance, which carries a $1/e(X)$ factor. First, as $\hat e(X) \to 0$ (or $\to 1$) a single unit's weight $1/\hat e$ explodes and can dominate the whole average — exactly where the overlap assumption was doing real work, since the survey variance is finite only when inclusion probabilities stay away from zero. The fix is to clip the propensity into a safe interval before inverting, $\hat e \leftarrow \min(\max(\hat e, c), 1-c)$, capping every weight at $1/c$. With the floor $c = 0.05$, clipping to $[0.05, 0.95]$, the maximum inverse-propensity weight is $1/0.05 = 20$. This trades a small bias on the handful of extreme units — where the data are thinnest and least trustworthy anyway — for a large variance reduction. Second, although $E[T/e(X)] = 1$ so the treated weights "should" sum to $N$, in any finite sample the realized arm weight totals drift, and that drift feeds straight in as noise. Self-normalizing — dividing each arm's weighted outcome sum by its own realized weight sum — kills that source of variance, giving the Hájek form
$$\hat\tau_{\text{norm}} = \frac{\sum_i T_i Y_i / \hat e(X_i)}{\sum_i T_i / \hat e(X_i)} - \frac{\sum_i (1-T_i) Y_i / (1-\hat e(X_i))}{\sum_i (1-T_i) / (1-\hat e(X_i))},$$
in which each arm's normalized weights sum to exactly one, so each arm estimate is a genuine weighted average of observed outcomes — which the raw Horvitz–Thompson form is not guaranteed to be. With a nonparametric $\hat e$ this normalized estimator attains the semiparametric efficiency bound (Hirano, Imbens & Ridder 2003): plugging in the estimated propensity is more efficient than using the true one, because estimating $\hat e$ soaks up exactly the right residual variation.

The harder target is the smooth function $\tau(x)$, and the same identity pays off pointwise. The per-unit quantity
$$\psi_i = \frac{T_i Y_i}{e(X_i)} - \frac{(1-T_i) Y_i}{1-e(X_i)}$$
satisfies, by the same conditional computation stopped before integrating out $X$, $E[\psi \mid X=x] = \mu_1(x) - \mu_0(x) = \tau(x)$. So $\psi$ is a noisy but conditionally-unbiased label for $\tau(x)$. I compute $\psi_i$ for every unit from the data and the fitted $\hat e$, then regress $\psi$ on $X$ with a flexible regressor; its conditional-mean fit estimates $E[\psi \mid X] = \tau(x)$, and averaging the predictions recovers the ATE consistent with the scalar estimator. This turns the unobservable per-unit effect into an ordinary regression with an observable, pointwise-unbiased pseudo-outcome — confounding removed by the weighting, heterogeneity recovered by the smoothing, no binning anywhere. The pieces line up against the failure modes: confounding is handled by the $e(X)$ cancellation that makes $\psi$ conditionally unbiased without any response surface; heterogeneity by regressing the pseudo-outcome on $X$; extreme weights by clipping into $[0.05, 0.95]$; residual weight noise by normalization for the scalar and the clip for the pseudo-outcome. The one genuine vulnerability is that everything rests on $\hat e$ being reasonable — a badly wrong propensity model leaves the cancellation imperfect — but for problems where assignment is easier to model than the outcome, or where one simply wants a method that uses the assignment mechanism the regression baselines ignore, that is the right trade, and it is a single point of failure that can be diagnosed by checking the fitted $\hat e$ and its overlap. I end with one modeling burden instead of two.

```python
import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


class CATEEstimator(BaseCATEEstimator):
    """IPW-based CATE estimator with propensity-score weighting.

    1. Estimate propensity e(X) = P(T=1|X) with a probability classifier.
    2. Clip e into [0.05, 0.95] for overlap / weight stability.
    3. Form the Horvitz-Thompson IPW pseudo-outcome
         psi = T*Y/e(X) - (1-T)*Y/(1-e(X)),   with  E[psi | X=x] = tau(x).
    4. Regress X -> psi; the conditional mean of psi is tau(x).
    """

    def __init__(self):
        seed = int(os.environ.get("SEED", "42"))
        self._prop_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=seed,
        )
        self._outcome_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=seed + 1,
        )

    def fit(self, X, T, Y):
        # e(X) = P(T=1|X)
        self._prop_model.fit(X, T)
        e_hat = self._prop_model.predict_proba(X)[:, 1]
        e_hat = np.clip(e_hat, 0.05, 0.95)            # caps every weight at 20

        # Horvitz-Thompson IPW pseudo-outcome; E[psi | X=x] = tau(x)
        pseudo_outcome = T * Y / e_hat - (1 - T) * Y / (1 - e_hat)

        # the conditional mean of the pseudo-outcome is tau(x)
        self._outcome_model.fit(X, pseudo_outcome)
        return self

    def predict(self, X):
        return self._outcome_model.predict(X)
```

The same fitted propensity gives the population-level Horvitz–Thompson / Hájek ATE that the pointwise estimator is built from, useful when only the scalar ATE is wanted:

```python
import numpy as np


def ipw_ate(X, T, Y, prop_model, clip=0.05, normalized=True):
    """Inverse-propensity-weighted average treatment effect."""
    prop_model.fit(X, T)
    e = np.clip(prop_model.predict_proba(X)[:, 1], clip, 1 - clip)
    w1 = T / e                     # treated:  inclusion prob e(X)
    w0 = (1 - T) / (1 - e)         # control:  "inclusion prob" 1 - e(X)
    if normalized:                 # Hájek self-normalized
        return np.sum(w1 * Y) / np.sum(w1) - np.sum(w0 * Y) / np.sum(w0)
    return np.mean(w1 * Y - w0 * Y)   # raw Horvitz-Thompson
```
