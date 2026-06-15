# Context: filling missing entries in a tabular dataset

## Research question

A data table `X` of shape `(n_samples, n_features)` arrives with some entries missing —
encoded as `NaN`. The downstream tools do not tolerate this. Most estimators require a
genuine vector space: classical regression in standard software silently drops every row
that has any missing entry, gradient-based learners cannot even represent `NaN`, and a
distance or kernel between two rows is undefined when one coordinate is absent. So before
any modelling can happen the table must be turned into a finite-valued table of the same
shape, with no `NaN`.

The precise goal is a *completion procedure*: a map from a table-with-`NaN`s to a
finite-valued table, with three properties. (1) It must be **learnable from the observed
entries alone** — there is no oracle for the missing values, and the procedure may not look
at the prediction target during fitting. (2) It must be **applicable out of sample**: a new
row arriving at test time, itself possibly carrying `NaN`s, has to be completed by the *same*
rule, because train and test are drawn from the same distribution and a procedure that
re-derives a different filling on the test set is not a well-defined predictor. (3) It must
not, by the act of filling, **inject a spurious signal** that the rest of the analysis will
mistake for real structure. What completes a table is not free: every value written into a
`NaN` slot is a guess, and the guess changes the statistics of the column it lands in.
Closing the gap between "the table has holes" and "a learner can consume it" — cheaply,
reproducibly, and without manufacturing structure — is the problem.

## Background

**The missingness mechanism decides what is even possible.** Rubin (1976) introduced the
controlling framework. Write `M` for the matrix of missingness indicators (`1` where an entry
is absent). The mechanism is classified by how `M` depends on the data:

- **Missing Completely At Random (MCAR):** `P(M | X) = P(M)`. Missingness is independent of
  every value, observed or not. The observed entries are then an unbiased sample of the full
  population, and simply discarding incomplete cases does not bias inferences.
- **Missing At Random (MAR):** `P(M | X)` depends only on the *observed* part of `X`. Under
  MAR (with the mechanism's parameter distinct from the parameter of interest — "ignorability")
  likelihood and Bayesian inference may ignore the missingness process.
- **Missing Not At Random (MNAR):** `P(M | X)` depends on the missing values themselves
  (e.g. high earners more likely to refuse the income question). The mechanism is then
  informative and must be modelled, or the resulting bias accepted.

This typology is load-bearing: it says *when* a filling can be innocuous (MCAR / MAR with
the missingness non-informative) and *when* missingness itself can carry information, forcing
the mechanism to be modelled or the mask to be retained.

**A single filled value is a guess treated as a fact.** Any procedure that writes one number
into each hole and proceeds as though it were observed pretends to a certainty it does not
have. The general consequence, known well before any particular filler is chosen, is that
estimates of spread computed on the completed table tend to come out *too small*, and that
standard errors of downstream estimates are correspondingly understated, because the
imputation uncertainty is never propagated.

**Inference vs. prediction are different objectives.** The classical missing-data literature
is *inferential*: it cares about unbiased estimates of means, variances, covariances, and
their standard errors. A *supervised-learning* setting cares instead about minimizing
prediction risk `E[ℓ(f(X), Y)]` on out-of-sample rows. A completion that damages the
inferential statistics of a column need not damage — and may be entirely adequate for — the
predictive use of that column, because the predictor is re-fit on the completed table and can
adapt to whatever the completion did. Holding this distinction clearly is part of the
landscape: a method judged harshly by the inferential yardstick can still be the right tool
under the predictive one.

**The cost of refusing to fill.** With `d` columns each independently missing at rate `p`
under MCAR, the probability that a given row is completely observed is `(1 − p)^d`. At
`p = 0.2` this is about `0.17` for `d = 8`, `0.055` for `d = 13`, and `0.0012` for `d = 30`:
restricting to complete rows can discard almost the entire table as the width grows. And even
a discard strategy leaves the out-of-sample problem unsolved — a test row with a hole still
cannot be scored. These facts define the pressure any completion procedure faces.

## Baselines

The prior approaches a completion procedure would be measured against, and react to.

**Complete-case analysis (listwise deletion).** Keep only the rows with no missing entry;
run the ordinary analysis on them. Core idea: sidestep the problem by restricting to the
fully observed subtable. Under MCAR it is unbiased. **Limitations:** it can delete the large
majority of the data once missingness is spread across many columns (the `(1 − p)^d` collapse
above); under MAR/MNAR the retained rows differ systematically from the deleted ones, biasing
the result; and it does not address test-time missingness at all, so it is not a usable
predictor when new rows can have holes.

**Available-case (pairwise) analysis.** For each statistic, use whatever rows have the
entries that statistic needs (e.g. estimate each covariance from the rows observing that
pair). Core idea: discard less than listwise deletion by varying the subsample per quantity.
**Limitations:** different statistics rest on different subsamples, so the pieces need not be
mutually consistent (a covariance matrix assembled pairwise can fail to be positive
semidefinite); like complete-case it is biased under non-MCAR mechanisms; and it produces no
completed table at all, so it does not feed a generic learner.

**Conditional / regression completion.** Fill a hole in a variable with a value *predicted
from the other observed variables* — e.g. regress the partially observed variable on the
fully observed ones over the rows where it is present, then substitute the fitted value for
the missing entries. Core idea: use the inter-variable relationships to make a better-targeted
guess than a single global number. **Limitations:** it needs a fitted predictive model per
incomplete variable (more machinery, more assumptions about functional form), and, by placing
every filled point exactly on the fitted surface, it still understates the spread of the
column — the filled values carry none of the residual scatter that real observations would.

**Stochastic regression completion.** As above, but add a random residual drawn to match the
estimated noise, so the filled points scatter around the fitted surface instead of lying on
it. Core idea: restore the variance the deterministic version removes. **Limitations:** it
introduces randomness (a single draw is one of many possible completions), it is less
*efficient* than a deterministic filler for estimating a mean, and a single such draw still
understates standard errors because it does not account for uncertainty in the fitted model
itself. The principled inferential fix — averaging over many draws — is heavier still.

## Evaluation settings

The natural yardsticks for a completion procedure, all available before one is chosen:

- **Tabular classification / regression datasets** with a controlled fraction of entries
  blanked out — for instance medium-width tables of a few hundred to a few thousand rows and
  on the order of ten to thirty numeric features, spanning binary classification,
  multi-class, and regression. Holding out *known* values as artificially missing makes the
  ground truth available for scoring the completion.
- **Missingness injected as 20% MCAR** (each cell independently blanked with probability
  0.2), so the mechanism is known and the simplest assumptions hold.
- **Reconstruction error** on the held-out (masked) entries: the root-mean-squared error
  between the filled value and the true value, computed only over the positions that were
  blanked. Lower is better.
- **Downstream predictive score**: fit a standard supervised model (e.g. a gradient-boosted
  tree) on the *completed* training table and measure its quality on a completed test table —
  accuracy for classification, `R²` for regression. This separates "did the filling reproduce
  the value" from "did the filling preserve what a learner needs," and it is the yardstick
  under which the inference-vs-prediction distinction above actually bites.
- Protocol: fit the completion on the training rows only (no peeking at test rows or at the
  target), then apply the *same* learned completion to the test rows before scoring.

## Code framework

The completion procedure plugs into the standard scikit-learn transformer protocol that the
benchmark already uses: a class with `fit(X, y=None)` that learns whatever it needs from the
training table (and must not touch the target `y` beyond ignoring it), `transform(X)` that
returns a same-shape table with every `NaN` replaced by a finite value, and the inherited
`fit_transform`. The data pipeline, the masking that creates the missing entries, the
downstream estimator, and the metrics all already exist; the single empty slot is the rule
that turns the observed entries of `X` into the numbers written back into its holes.

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

The harness supplies the table with holes and scores the completed output; `fit`/`transform`
are where the learned completion rule will live.
