# Context: estimating heterogeneous treatment effects from observational data (mid-2010s)

## Research question

We have data on a set of units — patients, voters, customers — described by a covariate
vector `X in R^d`, each of which either received a binary treatment (`W = 1`) or did not
(`W = 0`), and for each we observe a single outcome `Y`. We want to know not just whether
the treatment helps on average, but *for whom* it helps and by how much: the effect of the
treatment as a function of the unit's covariates. Formally the target is the Conditional
Average Treatment Effect

```
tau(x) = E[Y(1) - Y(0) | X = x],
```

the expected difference, at covariate value `x`, between the outcome a unit would have
under treatment, `Y(1)`, and the outcome it would have under control, `Y(0)`. Estimating
`tau` lets one personalize treatment regimes (treat the units the treatment actually
helps) and probe mechanisms (read off which covariates modulate the effect).

The difficulty is structural, not statistical. For each unit we ever observe only *one* of
its two potential outcomes — `Y(W_i)` — and never the counterfactual one, so the
unit-level effect `D_i = Y_i(1) - Y_i(0)` is never directly seen. In an *observational*
study treatment is not randomized: units that got the treatment may differ systematically
in `X` from those that did not, so a raw treated-minus-control comparison conflates the
effect with pre-existing differences (confounding). And the effect itself is *heterogeneous*
— it can vary across the covariate space in arbitrary, nonlinear ways — so the quantity to
be estimated is a whole function, not a single number. The question is how to turn noisy,
partially observed outcomes under an assignment that depends on covariates into an estimate
of the entire effect surface `tau(x)`.

## Background

The conceptual scaffolding is the Neyman–Rubin potential-outcomes model
(Splawa-Neyman 1923/1990; Rubin 1974). Each unit carries a pair `(Y_i(0), Y_i(1))` fixed
before assignment; assignment `W_i` selects which one becomes the observed
`Y_i^obs = Y_i(W_i)`; the other is missing. Drawing `(Y(0), Y(1), X, W)` from a
superpopulation `P`, the average treatment effect is `ATE = E[Y(1) - Y(0)]`, and the
covariate-conditional response surfaces are

```
mu_0(x) = E[Y(0) | X = x],   mu_1(x) = E[Y(1) | X = x].
```

A clean way to write the generating process is `X ~ Lambda`, `W ~ Bernoulli(e(X))`,
`Y(w) = mu_w(X) + eps(w)` with mean-zero noise `eps(w)` independent of `X, W`, where
`e(x) = P(W = 1 | X = x)` is the propensity score (Rosenbaum & Rubin 1983).

Two facts about what can be identified bound the whole enterprise. First, the unit-level
effect `D_i` is *not identifiable* without strong extra assumptions: one can build two
data-generating processes with identical observed-data distributions but different `D_i`
(Heckman, Smith & Clements 1997) — e.g. a Rademacher control outcome `Y(0) = ±1` with
equal probability, and treatment outcome that is either `Y(1) = Y(0)` (so `D_i = 0`) or
`Y(1) = -Y(0)` (so `D_i in {-2, 2}`); the observed data look the same, yet `D_i` differs.
The conditional mean `tau(x)` is identifiable in both, however, and is the right target.
That `tau` and not `D` is the achievable goal is sharpened by an MSE decomposition: for a
prediction `tau_hat(x)` made for a fresh unit at covariate value `x` (equivalently, after
conditioning on the fitted estimator),

```
E[(D_i - tau_hat(x))^2 | X_i = x, tau_hat]
  = E[(D_i - tau(x))^2 | X_i = x] + (tau(x) - tau_hat(x))^2,
```

the first term being irreducible variation of the unit effect around its conditional mean.
The estimator minimizing the MSE for the (unobservable) ITE is therefore the same one
minimizing the MSE for the CATE, so the metric of record is the Expected Mean Squared
Error `EMSE(P, tau_hat) = E[(tau(X) - tau_hat(X))^2]`.

Second, identification of `tau` from observational data requires two assumptions, both
standard by this time. *Ignorability / unconfoundedness* (Rosenbaum & Rubin 1983):
`(eps(0), eps(1)) ⟂ W | X`, i.e. conditional on the covariates, who got treated is
unrelated to the potential outcomes — all confounders are measured. Under ignorability the
observable conditional mean equals the counterfactual response surface,
`E[Y^obs | X = x, W = w] = mu_w(x)`, which is the bridge that lets one estimate
counterfactual quantities from observed regressions at all. *Overlap / positivity*: there
exist `e_min, e_max` with `0 < e_min < e(x) < e_max < 1` for all `x` in the support, so
that both arms are represented everywhere and the conditional means are estimable across
the whole covariate space.

The other piece of background is the state of supervised learning. By the mid-2010s,
flexible nonparametric regressors — random forests (Breiman 2001), Bayesian Additive
Regression Trees (Chipman, George & McCulloch 2010), gradient boosting, neural networks —
were mature, well-tested, and excellent at predicting `E[Y | features]` from
high-dimensional tabular data with little hand-tuning and built-in regularization. They
were built to *predict outcomes*, not to estimate treatment effects: there was no notion
of a counterfactual in their objective. But they were the workhorses everyone trusted, and
any practical effect estimator would gain enormously from being able to reuse them as
black boxes rather than reinventing regularization, cross-validation, and high-dimensional
fitting from scratch.

Two empirical regularities about effect estimation, established in the prior literature,
frame what a good estimator must cope with. The first is that the response surfaces `mu_0`,
`mu_1` are often individually complicated, yet their *difference* `tau` is comparatively
simple — frequently near zero, sparse, or close to linear — because the same covariates
that drive the baseline outcome cancel in the contrast (Kalla & Broockman; Sekhon &
Shem-Tov). The second is the unbalanced-design regularity: control data is often vastly
more plentiful than treated data, because untreated outcomes accrue passively from
administrative records, electronic medical records, and online platforms, while treated
units are scarce and expensive — so `m = #control` can be orders of magnitude larger than
`n = #treated`.

A standard way to grade nonparametric estimators rounds out the background: minimax rates
over families of distributions (Stone 1982; Györfi et al. 2006; Tsybakov 2009). For a
family `F` one asks the best worst-case EMSE achievable; a parametric (e.g. linear) family
admits the rate `N^{-1}`, while Lipschitz regression admits the nonparametric rate
`N^{-2/(2+d)}` in `d` dimensions (the one-dimensional case is `N^{-2/3}`, achieved by
Nadaraya–Watson or k-NN). Writing `S(a)` for the families whose minimax rate is at most
`N^{-a}`, one can grade a treatment-effect family by two exponents: `a_mu`, the rate at
which the response surfaces can be estimated, and `a_tau`, the rate at which the effect
surface can be estimated. The first empirical regularity above says `a_tau` is often
*larger* than `a_mu` — the effect is smoother than the outcomes.

## Baselines

**Difference of two arm-specific outcome models (the two-model / "two-tree" estimator;
Foster 2013; Athey & Imbens 2015).** Estimate the control response on the controls and the
treated response on the treated, with two independent regressions (possibly two different
base learners),

```
mu_hat_0 = M_0(Y ~ X)  on  {(X_i, Y_i) : W_i = 0},
mu_hat_1 = M_1(Y ~ X)  on  {(X_i, Y_i) : W_i = 1},
tau_hat(x) = mu_hat_1(x) - mu_hat_0(x).
```

Because the two arms are fit by two different models, the treatment can never be lost: it
is *which* model is used. Under ignorability and overlap each arm-model targets the right
counterfactual surface, and the EMSE is controlled by the two response-fit errors,
`EMSE(tau_hat) <= 2·(mu_1-error) + 2·(mu_0-error)`.

**Outcome regression / g-computation with a flexible Bayesian model (Hill 2011; Green &
Kern 2012).** Fit a single flexible model of the outcome on the covariates and a treatment
indicator with a nonparametric Bayesian regressor (BART), giving a fitted response surface
`f(x, w)`, and read the effect off as the difference of that surface across the indicator,
`f(x, 1) - f(x, 0)`, at fixed `x`. BART supplies coherent uncertainty intervals, needs
little tuning, and handles many predictors.

**Propensity reweighting (IPW; Rosenbaum & Rubin 1983).** Estimate `e_hat(x) = P(W=1|X=x)`
and reweight outcomes by the inverse propensity to recreate a pseudo-randomized comparison.

**Subgroup average effects (Hansen & Bowers 2009).** Partition the covariate space into
meaningful subgroups and report the ATE within each.

**Modified-splitting forest (Causal Forest; Wager & Athey 2017).** A random forest whose
splitting criterion is changed to target treatment-effect heterogeneity directly rather
than outcome prediction.

## Evaluation settings

The natural yardsticks are synthetic data-generating processes with known ground-truth
effects (so the estimation error is measurable) plus real field experiments.

- **Synthetic benchmarks** drawn from a known `P`: sample `X ~ Lambda`, assign
  `W ~ Bernoulli(e(X))`, draw `Y(w) = mu_w(X) + eps(w)`, and reveal only `Y^obs = Y(W)`.
  Vary the regime: a *simple / zero* effect surface (`tau ≈ 0` while the responses are
  complex), a *complex* effect surface, and an *unbalanced* design (`m >> n`). Families
  range over `X` dimensions from a handful to tens of covariates and sample sizes from
  hundreds to thousands. The known `tau(x)` lets one compute the estimation error directly.
- **Metrics** (both lower-is-better): the root-mean-square error of the effect surface,
  PEHE `= sqrt(mean((tau_hat - tau_true)^2))` (the empirical EMSE), and the absolute error
  of the average effect, `|mean(tau_hat) - ATE_true|`.
- **Protocol**: repeated cross-fitting / cross-validation over several random seeds, so the
  estimator is judged on stability across train/test splits rather than on one lucky
  realization.
- **Field experiments** with heterogeneous effects: a get-out-the-vote mailer experiment
  (Gerber, Green & Larimer 2008) and a door-to-door canvassing experiment on reducing
  prejudice (Broockman & Kalla 2016), used to see whether an estimator recovers
  interpretable heterogeneity on real data.

## Code framework

The estimator plugs into a fixed `fit` / `predict` interface used to grade every CATE
method. The harness owns nothing causal beyond the data triple `(X, W, Y)` and a slot for
a base supervised regressor; the existing primitives are just the scikit-learn-style
regressor (a `.fit(features, target)` / `.predict(features)` object), numpy array
plumbing, and the evaluation loop that calls `fit(X, W, Y)` then `predict(X_test)` and
scores `tau_hat` against the known `tau`. How the observed triple should be turned into
fitted regression problem(s), and how those fits should be combined into an effect at a
point, is exactly what is to be designed — so that is left as one empty slot.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class CATEEstimator:
    """Estimate the conditional average treatment effect tau(x) = E[Y(1)-Y(0) | X=x]
    from observational data (X, W, Y) with binary treatment W, reusing an ordinary
    supervised regressor as the workhorse. fit/predict interface graded by the harness."""

    def __init__(self):
        # an off-the-shelf, regularized nonparametric regressor is available as the
        # base learner; how many to fit, and on what data, is the open design choice
        self._base_learner = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8,
        )

    def fit(self, X, W, Y):
        # X: (N, d) covariates,  W: (N,) binary treatment,  Y: (N,) observed outcome.
        # TODO: turn (X, W, Y) into the regression problem(s) we will design,
        #       and fit the base learner(s) accordingly.
        raise NotImplementedError

    def predict(self, X):
        # Return tau_hat, one estimated treatment effect per row of X.
        # TODO: combine the fitted model(s) into an effect at each x.
        raise NotImplementedError


# existing evaluation harness the estimator plugs into
def evaluate(estimator, X_tr, W_tr, Y_tr, X_te, tau_true):
    estimator.fit(X_tr, W_tr, Y_tr)            # learn from the observed triple
    tau_hat = estimator.predict(X_te)          # predict the effect surface on test points
    pehe = np.sqrt(np.mean((tau_hat - tau_true) ** 2))
    ate_err = np.abs(np.mean(tau_hat) - np.mean(tau_true))
    return pehe, ate_err
```

The harness supplies `(X, W, Y)` and a base regressor; `fit`/`predict` is where the
construction of the regression problem and the effect rule will live.
