# Inverse Probability Weighting (IPW)

Inverse Probability Weighting estimates causal treatment effects from observational data by
reweighting each unit by the inverse of its probability of receiving the treatment it actually
got. It is the Horvitz–Thompson survey-sampling estimator carried over to treatment effects via
the propensity score: instead of modeling the outcome surface, it models the *assignment*
mechanism `e(x) = P(T=1|X=x)` and uses it to undo the confounding by reweighting. The treated
units are a non-representative sample of the population (drawn with probability `e(X)`); dividing
their outcomes by `e(X)` repairs that bias, and the `e(X)` cancels exactly in expectation.

## Problem it solves

From observational data `(X, T, Y)` — covariates `X`, binary treatment `T`, outcome `Y`, with
`Y = T·Y(1) + (1-T)·Y(0)` — estimate the average treatment effect `tau = E[Y(1)-Y(0)]` and the
conditional effect `tau(x) = E[Y(1)-Y(0)|X=x]`. Treatment was not randomized, so the naive
contrast `E[Y|T=1] - E[Y|T=0]` is confounded: the treated and control groups differ in `X`.

## Assumptions

- **Unconfoundedness:** `(Y(0), Y(1)) ⊥ T | X` — within a covariate cell, treatment is
  as-good-as-random.
- **Overlap (positivity):** `0 < e(x) < 1` for all `x` — every unit could have received either
  treatment. (Both due to Rosenbaum & Rubin 1983; Rubin 1978.)

## Key idea

Define the **propensity score** `e(x) = Pr(T=1|X=x) = E[T|X=x]`. Rosenbaum & Rubin (1983) show
it is a **balancing score** — `X ⊥ T | e(X)` — and the *coarsest* one (any balancing score
`b(x)` satisfies `e(x) = f(b(x))`), so adjusting for the scalar `e(X)` removes the same bias as
adjusting for all of `X`, and unconfoundedness holds given `e(X)`.

The treated units are a sample of the population drawn with inclusion probability `e(X)`. By the
**Horvitz–Thompson** unbiased-estimation argument (1952), the unique unbiased linear reweighting
of a sample drawn with inclusion probability `P(u_i)` weights each unit by `1/P(u_i)`: for
`sum_{i in sample} beta_i x_i` to be unbiased for `sum_i x_i` for every population vector, the
expectation `sum_i P(u_i) beta_i x_i` must match term by term, so `P(u_i) beta_i = 1` and
`beta_i = 1/P(u_i)`. Applied here, weighting treated outcomes by `1/e(X)` and control outcomes
by `1/(1-e(X))` recovers the population means of the potential outcomes:

```
E[ T·Y / e(X) ]        = E[Y(1)],     E[ (1-T)·Y / (1-e(X)) ] = E[Y(0)],
```

derived by conditioning on `X` and using unconfoundedness:
`E[T·Y/e(X) | X] = E[T·Y(1)/e(X) | X] = (1/e(X))·E[T|X]·E[Y(1)|X] = (1/e(X))·e(X)·mu_1(X) = mu_1(X)`,
then averaging over `X`. Hence

```
tau = E[ T·Y/e(X) - (1-T)·Y/(1-e(X)) ].
```

The `e(X)` in the denominator cancels the over-/under-representation that caused the confounding.
No outcome model `mu_w(x) = E[Y|T=w,X=x]` is needed.

## Estimators

`e(x)` is unknown, so estimate it with a probability classifier `ê(x)` (logit, as Rosenbaum &
Rubin recommend, or a flexible classifier when assignment is nonlinear).

**Horvitz–Thompson (vanilla) ATE:**

```
tau_hat = (1/N) sum_i [ T_i·Y_i/ê(X_i) - (1-T_i)·Y_i/(1-ê(X_i)) ].
```

**Self-normalized (Hájek) ATE** — divide each arm by its own realized weight sum:

```
tau_hat_norm =  [ sum_i T_i·Y_i/ê(X_i) ] / [ sum_i T_i/ê(X_i) ]
              - [ sum_i (1-T_i)·Y_i/(1-ê(X_i)) ] / [ sum_i (1-T_i)/(1-ê(X_i)) ].
```

The normalized weights in each arm sum to exactly 1, so each arm is a genuine weighted average
of observed outcomes, typically with lower variance. With a nonparametric `ê`, the normalized estimator
attains the semiparametric efficiency bound (Hirano, Imbens & Ridder 2003) — weighting by the
*estimated* propensity is more efficient than weighting by the true one.

## CATE via the IPW pseudo-outcome

Read the same identity *conditionally* (without integrating out `X`):

```
psi_i = T_i·Y_i/e(X_i) - (1-T_i)·Y_i/(1-e(X_i)),     E[psi | X=x] = mu_1(x) - mu_0(x) = tau(x).
```

So `psi` is a pointwise-unbiased (noisy) label for the conditional effect. Regressing `psi` on
`X` with any flexible regressor recovers `tau(x)` as the conditional mean; averaging its
predictions recovers the ATE.

## Overlap and clipping

The Horvitz–Thompson variance carries a `1/e(X)` factor, so as `ê(X) -> 0` (or `1`) a single
unit's weight `1/ê` explodes and dominates the estimate — this is where the overlap assumption
bites (H–T requires `P(u_i) > 0`). **Clip** `ê` into `[0.05, 0.95]` before inverting: this caps
every weight at `1/0.05 = 20`, trading a small bias on the extreme units (where data are thinnest
anyway) for a large variance reduction. The implementation uses this `0.05` floor.

## Working code

CATE estimator (fills the `CATEEstimator` slot of the harness: fit propensity → clip → IPW
pseudo-outcome → regress on `X`):

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

Population-level Horvitz–Thompson / Hájek ATE (the scalar estimator the pointwise one is built
from):

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

## Relation to prior methods

- **Regression / S- and T-learners** model the outcome surface(s) `mu_w(x)` and ignore the
  assignment mechanism; IPW does the opposite — models only `e(x)` and never the outcome.
  Consistency requires the propensity model to be correct (vs. the outcome model for regression).
- **Subclassification / matching on `e(x)`** use the propensity score coarsely (bin or pair);
  IPW uses it as a smooth continuous weight and yields a closed-form estimate, and the
  pseudo-outcome version yields a smooth `tau(x)`.
