## Research question

A data table `X` of shape `(n_samples, n_features)` arrives with some entries missing,
encoded as `NaN`. The downstream tools do not tolerate this: a regression routine silently
drops any row with a hole, a gradient-based learner cannot put a `NaN` in a tensor, and a
distance or kernel between two rows is undefined the moment one coordinate is absent. So
before any modelling can happen the table must be turned into a finite-valued table of the
same shape, with no `NaN`.

The goal is a *completion procedure* that (1) learns from the observed entries alone — there
is no oracle for the missing values, and it may not look at the prediction target while
fitting — and (2) is applicable out of sample: a new row at test time, itself possibly
carrying holes, must be completed by the *same* frozen rule. How should missing entries be
filled so that the same learned rule applies both on the training table and on new rows?

## Background

**The missingness mechanism decides what is possible.** Rubin (1976) introduced the
controlling framework. Write `M` for the matrix of missingness indicators (`1` where an entry
is absent). Under **Missing Completely At Random (MCAR)**, `P(M | X) = P(M)`: missingness is
independent of every value, so the observed entries are an unbiased sample of the population
and a hole is a coin flip that landed tails — nothing about the value made it disappear.
Under **Missing At Random (MAR)**, `P(M | X)` depends only on the *observed* part of `X`, so
conditioning on what is seen the holes are again uninformative about themselves. Under
**Missing Not At Random (MNAR)**, the probability of being missing depends on the missing
value itself, and the mask then carries information no observed value can reconstruct. The
clean regime — where putting a number into a hole can be harmless — is MCAR / MAR with the
mechanism non-informative.

**Nearest-neighbor estimation.** The oldest non-parametric idea for "guess the value at a
point from nearby points" is the nearest-neighbor rule (Cover & Hart, 1967): classify or
predict a query by the label/value of its closest training example under a distance, with the
asymptotic risk of the 1-NN rule bounded by twice the Bayes risk. Averaging the `K` closest
instead of taking the single closest trades a little bias for much less variance, and turns
the prediction into a *local average* of the response surface — exactly the kind of estimate
that respects local structure without committing to a global functional form. The whole rule
rests on a distance: "nearby" has to mean something.

**A distance is undefined when a coordinate is missing.** The squared Euclidean distance
between two rows, `Σ_ℓ (x_ℓ − y_ℓ)²`, needs every coordinate of both rows. If either row has
a hole at coordinate `ℓ`, the term `(x_ℓ − y_ℓ)²` cannot be formed. Dixon (1979), studying
pattern recognition with partly missing data, addressed this with the **partial distance
strategy**: form the squared distance over only the coordinates *present in both* rows, then
*scale it up* by the ratio of the total number of coordinates to the number of present ones,
to estimate what the full-dimensional squared distance would have been. The scale-up is
load-bearing — without it, a pair that happens to share more observed coordinates accumulates
more squared terms and looks systematically farther than a pair sharing fewer, so the count
of overlap, rather than genuine similarity, would drive neighbor selection.

**Distance is sensitive to scale and to outliers.** Euclidean distance squares each
coordinate difference, so a single coordinate with a large spread, or a single wild value,
dominates the sum and can swamp the contribution of every other coordinate. In domains where
the raw measurements span orders of magnitude and carry occasional extreme values, this makes
a naive Euclidean distance fragile. A standard remedy is to **log-transform** heavy-tailed
positive measurements: the log compresses large values, pulls in the tail, and brings the
coordinates onto a comparable scale, so that a sum-of-squares distance is no longer hostage to
a few outliers. Whether a sum-of-squares distance is even appropriate is therefore conditional
on the data being on a sensible scale.

**Single-imputation understates spread.** Any procedure that writes one number into each hole
and proceeds as though it were observed pretends to a certainty it does not have; the general
consequence is that variances on the completed table come out too small and downstream standard
errors are understated, because the imputation uncertainty is never propagated. This matters
for *inference*; under a *predictive* objective, where the downstream model is re-fit on the
completed table, the relevant question is instead whether the filled values are accurate and
consistent between train and test.

## Baselines

Useful baselines:

**Zero-fill.** Write `0` into every hole. Core idea: the cheapest possible placeholder.

**Mean imputation.** Replace each hole with a single constant for its feature (or, in the
gene-expression orientation, the row average for that gene). Core idea: the least-squares
constant predictor of a feature is its observed mean, so among no-covariate guesses the mean
is optimal, and it is the unique constant that leaves the feature mean unchanged. Cheap,
deterministic, trivially frozen and replayed on test rows.

**Global low-rank (eigen-vector) completion.** Learn a small set of `J` basis vectors from
the fully observed rows (a truncated SVD of the complete subtable), then for an incomplete row
regress its observed coordinates on those bases and read off the bases' values at the missing
coordinates; an all-data variant iterates this as an EM fit of a rank-`J` approximation that
ignores the missing entries in its Frobenius objective. Core idea: the table is approximately
low-rank, so a few global directions explain most of every row, and a hole can be recovered by
projecting onto them.

**Iterative regression / EM completion (chained equations).** For each feature with holes,
regress that feature on all the others over the rows where it is observed, predict the holes,
and iterate — initializing the holes with the column means and refining as the imputations
improve (an EM fit of a joint Gaussian, with the imputations as a by-product; generalizable
to non-linear per-feature regressors). Core idea: model each feature as a function of the
rest and cycle until the completed table is self-consistent.

## Evaluation settings

Natural yardsticks for a completion procedure:

- **Tabular classification / regression datasets** of a few hundred to a few thousand rows and
  on the order of ten to thirty numeric features, spanning binary classification, multi-class,
  and regression. Holding out *known* values as artificially missing makes the ground truth
  available for scoring.
- **Missingness injected as 20% MCAR** (each cell independently blanked with probability 0.2),
  so the mechanism is known and the simplest assumptions hold. Robustness is naturally probed
  by sweeping the missing fraction over a range such as 1-20%.
- **Reconstruction error** on the held-out (masked) entries — the root-mean-squared error
  between filled and true value, computed only over the blanked positions (often normalized by
  the spread of the data, as a normalized RMS error). Lower is better.
- **Downstream predictive score** — fit a standard supervised model (e.g. a gradient-boosted
  tree) on the *completed* training table and measure its quality on a completed test table:
  accuracy for classification, `R²` for regression.
- Protocol: fit the completion on the training rows only (no peeking at test rows or at the
  target), then apply the *same* learned completion to the test rows before scoring.

## Code framework

The completion procedure plugs into the standard scikit-learn transformer protocol: a class
with `fit(X, y=None)` that learns whatever it needs from the training table (and must not
touch the target `y`), `transform(X)` that returns a same-shape table with every `NaN`
replaced by a finite value, and the inherited `fit_transform`. The data pipeline supplies the
table with holes and scores the completed output; `fit`/`transform` are where the learned
completion rule lives.

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CustomImputer(BaseEstimator, TransformerMixin):
    """Completion transformer: maps a table with NaNs to a finite-valued table
    of the same shape. Learns from the observed entries of the training table
    in fit(); applies the SAME learned rule to any table in transform()."""

    def __init__(self, random_state=42, max_iter=10):
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y=None):
        # X: (n_samples, n_features) numpy array, NaN for missing.
        # Learn whatever the completion rule needs from the OBSERVED entries.
        # Must not use the prediction target.
        # TODO: the completion rule we will design — what to learn here.
        return self

    def transform(self, X):
        # X: (n_samples, n_features) numpy array with NaNs (train or test).
        # Return: same-shape array with every NaN replaced by a finite value.
        X_imputed = X.copy()
        # TODO: fill each NaN using what fit() learned.
        return X_imputed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
```
