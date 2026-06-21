# Context: imputing missing values in mixed-type tabular data (circa 2010-2011)

## Research question

A great many analysis methods — regression, clustering, PCA, classifiers, distance
computations — assume a fully observed data matrix, yet real data sets in biology and medicine
are riddled with missing entries. High-throughput measurement (microarrays, mass spectrometry,
clinical registries) routinely produces a matrix `X` of shape `n × p` with `NaN` scattered
through it, and the analyst must produce a finite-valued completion `X_imp` before anything
downstream can run. The goal is a *good* completion: one whose filled-in values are close to
the (unknown) truth, so the reconstruction error is small and a downstream model trained on
`X_imp` performs as if the data had been observed.

How should an imputer be designed to handle real tabular data, where continuous measurements
sit next to categorical ones, relationships among variables may be nonlinear, and the analyst
has no prior knowledge of which parametric family each variable's conditional distribution
belongs to?

## Background

The field had a clear taxonomy of imputation ideas.

**Single vs. multiple imputation, and the missingness mechanism.** Rubin (1978) framed
multiple imputation: rather than filling each hole with one number, draw several plausible
completions so that the *uncertainty* of the imputation can be propagated into downstream
inference. The standard taxonomy of why data are missing — missing completely at random
(MCAR), at random (MAR), not at random — governs what is recoverable; under MCAR the
missingness is independent of all values, the most benign case and the one used to *evaluate*
imputers by artificially deleting known entries and measuring how well they are recovered.

**Exploiting inter-feature correlation.** The simplest completion, replacing each missing
entry by its column mean, uses no other variable and so ignores all cross-feature structure;
it is the baseline everything must beat. The next idea up is to *predict* a missing entry from
the other variables, which is where the real structure lives. Two families do this:
neighborhood methods (find rows similar on the observed coordinates, average their values) and
regression methods (fit a model of the incomplete variable on the others).

**The chained-equations / regression-switching idea.** A central, load-bearing idea is to
avoid specifying a single joint model over all `p` variables — hard and often impossible for
mixed types — and instead specify, for each incomplete variable, a *conditional* model of that
variable given all the others. Imputation then becomes a round-robin: cycle through the
variables, each time regressing one on the rest (using the current best fill of the rest) and
replacing its holes with predictions, and repeat the cycle. Because each pass uses the most
recent imputations of the earlier variables, the procedure resembles a Gibbs sampler over the
collection of imputed values, and it converges in practice in surprisingly few cycles —
typically five to twenty — partly because the imputed values carry random noise that decorrelates
successive states and speeds mixing. The theoretical hitch is that a set of conditional models
need not correspond to any actual joint distribution (the "compatibility" question), though in
practice with modest missingness and well-fitting conditionals this is a minor concern.

**Random forests as a general-purpose predictor (Breiman 2001).** A random forest grows many
unpruned CART trees, each on a bootstrap resample of the rows, and at every node restricts the
split search to a random subset of `m_try` of the `p` variables; predictions are averaged
(regression) or voted (classification). CART splits handle continuous and categorical predictors
natively and are invariant to monotone rescaling of any variable. The tree structure captures
nonlinearities and interactions automatically. Breiman's analysis gives the mechanism: the
forest's generalization error is bounded as
`PE* ≤ ρ̄ (1 − s²)/s²` (classification) and `PE*(forest) ≤ ρ̄ · PE*(tree)` (regression), where
`s` is the strength of an individual tree and `ρ̄` is the average correlation between the
trees' errors; injecting randomness (bootstrap rows + a random `m_try`-subset of features per
node) *decorrelates* the trees, driving `ρ̄` down and hence the error down, while strength is
maintained — the empirical observation being that error is "insensitive to the number of
features selected to split each node," with one or two features near optimal. By the strong law
of large numbers the forest error converges as the number of trees grows, so adding trees never
overfits — it only costs runtime, which is roughly linear in the number of trees.

**Out-of-bag error, for free.** Because a size-`n` bootstrap sample omits any fixed row with
probability `(1 − 1/n)^n → 1/e ≈ 36.8%`, every row is out-of-bag for a little more than one
third of the trees. Predicting each row using only the trees that did not see it yields the
out-of-bag (OOB) prediction, and the OOB error is an internal, essentially unbiased estimate of
generalization error — "as accurate as using a test set of the same size" — so it removes the
need for a held-out set or cross-validation.

**Evaluation convention for imputation accuracy.** For continuous variables the accepted
yardstick (Oba et al. 2003, from Bayesian-PCA gene-expression imputation) is the *normalized*
root mean squared error, RMSE divided by the spread of the true values, so that 0 is perfect
and ≈ 1 is no better than the mean; for categorical variables the analogue is the proportion of
falsely classified entries. Errors are measured only over the artificially deleted entries
whose truth is known.

## Baselines

These are the prior methods a new imputer would be measured against.

**Mean (unconditional) imputation.** Replace each missing entry of a column by the mean of
that column's observed entries. It is the no-covariate predictor — it uses no other variable.

**KNNimpute (Troyanskaya et al. 2001).** For a variable with a missing entry, find its `k`
nearest neighbors (rows, in the gene-expression setting variables) by Euclidean distance over
the observed coordinates and impute a distance-weighted average of their values. The neighbor
count `k` is a tuning parameter; the Euclidean distance requires the variables to be put on a
common scale, so the data must be standardized (and de-standardized afterward).

**MICE / fully conditional specification (Van Buuren & Oudshoorn 1999; Van Buuren 2007; built
on Schafer 1997).** The chained-equations method: specify a parametric conditional model per
incomplete variable — linear regression with normal errors for continuous, logistic for binary,
polytomous logistic for multi-category — and iterate the round-robin described above, drawing
imputations from each conditional given the current fill of the others. It handles mixed types
and, via the multiple-imputation scheme, can express the uncertainty of each imputed value.

**MissPALasso (Städler & Bühlmann 2010).** An EM-type method for high-dimensional, near-normal
continuous matrices: regress the missing variables on the observed ones with an `ℓ1` (lasso,
Tibshirani 1996) penalty, then use the fitted coefficients to update the latent distribution in
the E-step. It carries a penalty parameter `λ` and requires the regressions to be standardized
to a common scale.

**Iterative soft-thresholded SVD / matrix completion (Mazumder, Hastie & Tibshirani 2010).**
For a single continuous matrix with missing entries, repeatedly fit a low-rank approximation by
soft-thresholding the singular values and refill the holes with the current low-rank estimate,
iterating to convergence. It imposes a global low-rank linear structure on the whole matrix.

## Evaluation settings

The natural yardsticks already in use for imputation accuracy, all using artificially deleted
entries whose truth is known so the recovery error can be measured.

- **Corruption protocol.** Take a complete data set, delete a fraction of entries completely at
  random (MCAR) — commonly swept at 10%, 20%, 30% — then impute and compare to the held-out
  truth. Repeat over many independent random deletions (e.g. 50 simulations) and average; a
  paired Wilcoxon signed-rank test compares per-simulation error rates between methods.
- **Metrics.** For continuous variables, the normalized RMSE
  `NRMSE = sqrt( mean((X_true − X_imp)²) / var(X_true) )` over the deleted continuous entries
  (≈0 good, ≈1 no better than the mean); for categorical variables, the proportion of falsely
  classified entries (PFC) over the deleted categorical entries.
- **Data sets.** A diverse battery spanning the regimes the method must survive: continuous-only
  gene-expression and biomedical-signal tables (tens to thousands of variables, including a
  `p ≫ n` case with `p ≈ 12 600`, `n ≈ 110`), categorical-only tables (binary SPECT features,
  4-letter DNA promoter sequences, multi-level lymphography codes), and mixed continuous +
  categorical tables (proteomics biomarkers, peptide-search variables, a clinical registry with
  48 continuous and 76 categorical variables).
- **Cost.** Runtime is reported alongside accuracy, since some baselines (penalized-regression
  EM, fully Bayesian MICE) are far slower than neighborhood methods and a usable imputer must be
  affordable at the data sizes above.

## Code framework

The imputer plugs into the standard transformer interface: a class with `fit(X)` that learns
whatever it needs from a table containing `NaN`, and `transform(X)` that returns a same-shape
table with the holes filled by finite values (no test labels are ever touched). The primitives
that already exist are a univariate initial fill (`SimpleImputer(strategy="mean")`), a
supervised estimator interface with `fit` and `predict`, and the generic *round-robin* shape
inherited from chained equations — start from a complete initial fill, then repeatedly sweep the
incomplete variables, each time fitting some per-variable predictor on the others and overwriting
that variable's holes, stopping when the sweeps stop changing the matrix or a cap is hit.

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class CustomImputer(BaseEstimator, TransformerMixin):
    """Round-robin (chained) imputer skeleton. Starts from a complete initial
    fill, then repeatedly re-imputes each variable with missing values by some
    per-variable predictor trained on the other variables, until the sweeps
    stop changing the matrix or a cap is hit. The per-variable predictor and
    the convergence test are not yet chosen."""

    def __init__(self, random_state=42, max_iter=10):
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y=None):
        # X: (n_samples, n_features) float array with NaN for missing entries.
        # Learn the imputation from X only (must not use test labels).
        self._run(X)
        return self

    def transform(self, X):
        # Return a same-shape array with every NaN replaced by a finite value.
        return self._run(X)

    def _run(self, X):
        X_imp = X.copy()
        n_samples, n_features = X_imp.shape

        # initial complete fill so every predictor block is defined from sweep 1
        col_mean = np.nanmean(X_imp, axis=0)
        nan_mask = np.isnan(X_imp)
        for j in range(n_features):
            X_imp[nan_mask[:, j], j] = col_mean[j]

        missing = np.isnan(X)                      # original missingness pattern
        # TODO: decide the order in which variables are re-imputed.

        for _ in range(self.max_iter):
            X_prev = X_imp.copy()
            for j in range(n_features):
                obs = ~missing[:, j]
                mis = missing[:, j]
                if mis.sum() == 0:
                    continue
                others = [k for k in range(n_features) if k != j]
                # TODO: the per-variable predictor we will choose. Fit it on the
                #       observed rows of variable j using the other variables as
                #       inputs, then overwrite j's missing entries with its
                #       predictions, using the current X_imp for the inputs.
                pass
            # TODO: decide whether the sweeps have converged, and which iterate
            #       to return.
        return X_imp
```
