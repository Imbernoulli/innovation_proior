# Context: imputing multivariate missing data

## Research question

We have a data table `Y` with `p` variables and `n` rows in which entries are
missing, scattered across more than one column. We want to fill those holes with
plausible values so that the completed table can be used for whatever analysis was
intended — fitting a downstream model, estimating a correlation, a regression
coefficient. Two things must hold for the fill to be any good. First, it must
exploit the *relationships between variables*: if cholesterol and BMI move
together, a missing cholesterol value should be informed by the row's BMI, not
just by the cholesterol column's average. Second — and this is the part that is
easy to forget — the fill must not pretend to know more than it does: imputed
values should carry the right amount of *scatter*, because a single best guess
plugged in as if observed shrinks variance and distorts the very correlations we
were trying to preserve.

The hard structural fact is that the missingness is *multivariate* and *tangled*.
The variable we want to predict, say `Y_j`, is best predicted from the other
variables `Y_{-j}` — but those other variables are themselves incomplete, and the
dependence is circular: `Y_1` would help impute `Y_2`, yet `Y_2` would help impute
`Y_1`. On top of that the columns are of mixed type — some binary, some ordered,
some continuous and skewed and bounded — and the relations among them can be
nonlinear or constrained (a sum that must equal its parts; a value that cannot go
negative). A solution has to cope with all of this on real tabular data, not on a
clean Gaussian toy. What it must achieve: completed values that (1) borrow strength
across the correlated columns, (2) respect the type and range of each variable,
and (3) reflect, rather than erase, the uncertainty about what the missing values
really were.

## Background

The starting point is *why* it is even legitimate to fill missing values from the
data we can see. Rubin (1976) settled this. Partition the data into its observed
part `Y_obs` and missing part `Y_mis`, and introduce the response indicator `R`
(which entries are present). The data are *missing completely at random* (MCAR) if
the probability of being missing does not depend on the data at all; *missing at
random* (MAR) if, conditional on `Y_obs`, it does not depend on `Y_mis`. Rubin
showed that under MAR, and provided the parameters of the data model and of the
missingness mechanism are distinct, the missingness mechanism is *ignorable* for
likelihood-based and Bayesian inference: one may model `P(Y_mis | Y_obs)` and
proceed as if the holes were not there. This is the licence the whole field
operates under — it says the conditional distribution of the missing given the
observed is the right object to draw imputations from. MCAR, where missingness is
unrelated to everything, is the strict special case, and it is the regime in which
the simplest fills are least harmful.

The second pillar is *multiple* imputation and Rubin's (1987) accounting for
uncertainty. The idea: do not produce one completed table but `m` of them, each
with the holes filled by a different plausible draw, analyze each completed table
with the complete-data method you would have used anyway, and then *pool*. For a
scalar estimand `Q`, with per-imputation estimates `Q_hat_h` and their sampling
variances `U_h` for `h = 1..m`, the pooled point estimate and variance are

```
Qbar = (1/m) * sum_h Q_hat_h
Ubar = (1/m) * sum_h U_h                            # within-imputation variance
B    = (1/(m-1)) * sum_h (Q_hat_h - Qbar)^2         # between-imputation variance
T    = Ubar + (1 + 1/m) * B                         # total variance
```

The between-imputation term `B` is the whole point: it is the extra variance caused
by *having* missing data, and it is exactly what a single deterministic fill throws
away. If every completed table is identical (one best guess), then `B = 0`, the
total variance collapses to the within-imputation part, confidence intervals come
out artificially narrow, and tests gain false precision. So a *valid* imputation
must produce draws that genuinely differ across the `m` tables, and they differ
only if the imputation injects the right randomness — both the uncertainty in the
estimated relationships and the residual scatter that even a perfect model leaves.

A standard tool for a single incomplete column, given the others as predictors, is
Bayesian linear regression with a draw from the posterior. Regress the observed
part of `Y_j` on the other variables; but instead of plugging in the least-squares
fit, draw a residual variance and a coefficient vector from their posteriors, then
form the imputation as the predicted mean *plus* a residual noise term. The two
draws are the two uncertainties Rubin's `B` needs: parameter uncertainty (we do not
know the regression exactly) and residual uncertainty (the model does not explain
everything). A practical wrinkle is regularization: with many predictors, small
samples, collinearity, or near-empty cells, the cross-product matrix `X'X` is
ill-conditioned, so the regression must be stabilized. MacKay's (1992) evidence
framework supplies a principled way to do this — put a Gaussian prior on the
coefficients with precision `lambda` and a noise precision `alpha`, center the data
so the intercept is handled outside the probabilistic weights, and set the two
precisions by maximizing the marginal likelihood; the resulting posterior mean is
a ridge solution `(lambda/alpha · I + X'X)^{-1} X'y` whose penalty is chosen
automatically from the data, with an "effective number of parameters" `gamma`
controlling the trade-off. This gives a fast, self-tuning per-column linear model
that does not blow up on collinear or small-`n` data.

Some empirical facts about the existing fills are worth pinning down, because they
motivate everything. Replacing a column's holes by its mean is known to be doubly
wrong: it adds zero scatter (so it shrinks the variance and pulls every pairwise
correlation toward zero), and it ignores the other variables entirely. A
single regression fill (predicted mean, no noise) fixes the second problem but not
the first — it still understates variance and, by placing every imputation exactly
on the regression surface, *inflates* correlations. Even a *stochastic* regression
that adds residual noise but uses the fixed least-squares coefficients is observed
to under-cover: its interval coverage comes out around 0.908 instead of 0.95,
because it omits the parameter-uncertainty draw. These are the diagnostic
observations the field already has in hand.

## Baselines

These are the prior approaches a new method would be measured against and would
react to.

**Mean / column-average imputation.** Replace every missing entry in a column with
the mean (or median) of that column's observed values. Trivial and fast. *Gap:* it
uses no information from the other variables, so it cannot recover any cross-feature
structure; and because every fill is the same constant, it adds no scatter — the
imputed column's variance shrinks and its correlations with other columns are
pulled toward zero, biasing essentially any downstream multivariate analysis.

**k-Nearest-Neighbours imputation (Troyanskaya et al., Bioinformatics 2001).** For
a row with a missing entry in column `j`, find the `k` rows most similar to it,
where similarity is measured (typically by Euclidean distance) on the features both
rows have observed; impute the missing entry as a distance-weighted average of
those `k` neighbours' values in column `j`. Default `k = 5`. This genuinely
exploits the data's local structure — rows that resemble each other in the observed
features are taken to resemble each other in the missing one — and it makes no
parametric assumption. *Gap:* it is one-shot and local. It never builds an explicit
model of how `Y_j` depends on `Y_{-j}`, so it cannot represent a global
relationship or weight predictors by their relevance; distances are computed on
whatever happens to be observed in each pair, which degrades as missingness rises;
and it does not iterate, so an early rough fill of one column is never revisited in
light of better fills of the others.

**Joint modeling (Schafer, *Analysis of Incomplete Multivariate Data*, 1997).**
Posit a single multivariate distribution for all `p` variables at once — most
commonly a multivariate normal with an unstructured covariance, or for categorical
data a log-linear or general-location model — and draw the missing entries from
their conditional distribution under that joint, via data-augmentation MCMC (the
NORM software). When the assumed joint is a reasonable description of the data this
is clean and statistically principled: the imputations are guaranteed to come from
a coherent probability model, and the parameter and residual uncertainty both fall
out of the MCMC. *Gap:* it stands or falls on the existence of a tractable joint.
Real tabular data mixes binary, ordinal, continuous-skewed, and bounded variables,
has nonlinear and censored relations, and obeys deterministic constraints — for
which no convenient multivariate distribution is available. Forcing a multivariate
normal onto such data produces imputations that ignore bounds and types (continuous
fills for a binary variable, negative fills for a positive one), and extending the
joint to honor every feature of the data quickly becomes intractable.

**Stochastic regression imputation (single column).** Regress an incomplete column
on the others (using the complete rows), and impute the predicted value plus a
random residual draw `z · sigma_hat`. This restores some scatter and uses the other
variables. *Gap:* it tackles only *one* incomplete column at a time and treats the
predictors as fully observed; with missingness in several columns at once there is
no recipe here for the circular dependence among them. And using the fixed
least-squares estimates omits the parameter-uncertainty draw, so as a building
block it under-covers (the ~0.908 coverage noted above) even when the single-column
model is correct.

## Evaluation settings

The natural yardsticks, all pre-existing:

- **Tabular datasets with introduced missingness.** Standard small-to-medium tables
  are standardized and then corrupted with **20% MCAR** missingness (each entry
  independently dropped with probability 0.2, never leaving an entirely empty row
  or column): a breast-cancer diagnostic table (569 rows, 30 continuous features,
  binary label), a wine table (178 rows, 13 features, 3-class label), and a
  California-housing table (5,000 rows, 8 features, continuous target). Standardized
  features mean a per-column scale on the order of unity.
- **Reconstruction error.** Root-mean-squared error between the imputed values and
  the true (held-out) values, computed only over the artificially masked entries —
  lower is better.
- **Downstream task performance.** Fit a gradient-boosting model on the *completed*
  table and measure its cross-validated accuracy (classification) or `R^2`
  (regression) — higher is better. This tests whether the imputation preserved the
  predictive signal, not just the marginal values.
- **Classical statistical yardstick.** In the broader literature, imputation is also
  judged by the bias and confidence-interval *coverage* of a pooled estimand under
  simulated MCAR/MAR — the 0.95-coverage criterion behind the observations above.

## Code framework

The imputer plugs into a standard scikit-learn transformer interface and the
benchmark's evaluation harness. The harness loads a table, standardizes it, drops
20% of entries to `NaN`, calls the imputer's `fit_transform`, and scores the
result. The primitives that already exist are the generic numerical stack: `numpy`
for arrays and `NaN` masks (`np.isnan`, `np.nanmean`), `scipy`, and the rest of
`scikit-learn` — including a mean/median fill (`SimpleImputer`), linear regressors,
nearest-neighbour utilities, and the `BaseEstimator` / `TransformerMixin` base
classes. What does *not* yet exist is the rule for turning a table full of `NaN`s,
with missingness scattered across many interdependent columns, into a finite,
relationship-respecting completed table — that rule is the single empty slot.

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CustomImputer(BaseEstimator, TransformerMixin):
    """Fill the NaN entries of a numeric table (n_samples x n_features).

    fit(X)        : learn whatever the imputation rule needs from X (X has NaNs)
    transform(X)  : return X with every NaN replaced by a finite value
    """

    def __init__(self, random_state=42, max_iter=30):
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y=None):
        # X: np.ndarray (n_samples, n_features), NaN marks missing entries.
        # TODO: the imputation procedure we will design — how the holes in
        #       interdependent columns get turned into finite values.
        return self

    def transform(self, X):
        # X: np.ndarray (n_samples, n_features) with NaNs.
        # Must return an array of the same shape with no NaNs.
        # TODO: produce the completed table.
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
```

The harness only ever calls `fit_transform(X)` on the `NaN`-laden table and then
measures reconstruction error and downstream score; everything about *how* the
holes are filled lives behind these stubs.
