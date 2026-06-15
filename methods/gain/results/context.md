# Context: imputing missing tabular data (circa 2017-2018)

## Research question

A great many real datasets arrive with holes in them. A clinical record may be missing a
patient's respiratory rate because it was never measured, or a biopsy-derived feature because the
procedure was too invasive to perform; a survey has non-responses; a sensor drops readings. The
practical task is to fill those holes — to produce, from a partially observed feature vector, a
completed vector whose filled-in entries are as faithful as possible to the values that would have
been there. Two things make this hard at once. First, the entries are not independent: a feature's
missing value is informative about, and informed by, the *other* features of the same record, and
a good filler has to exploit those dependencies rather than collapse each column to a single
summary. Second — and this is the constraint that rules out a whole class of methods — in many
settings missingness is part of the inherent structure of the problem, so a fully observed
("complete") version of the data does not exist *anywhere*, not even as a training set. Whatever
learns to impute has to learn from the incomplete data itself.

There is a deeper version of the goal than "guess each missing number." For a record with observed
part `x̃`, the principled object to recover is the *conditional distribution* `P(X | X̃ = x̃)` of the
complete vector given what was seen — not merely its mean. Modeling the distribution rather than
the point estimate is what lets one draw several different completions of the same record (multiple
imputation) and so represent the genuine uncertainty about the missing values, instead of pretending
a single number is certain. A method that recovers only the conditional expectation throws that
uncertainty away and tends to shrink imputations toward an average, distorting downstream analyses.
So the target is: a way to sample plausible completed vectors from `P(X | X̃)`, learned from data
that is itself incomplete, that respects cross-feature structure and works for mixed
continuous/binary features.

To make the theory tractable, the standard simplifying assumption is **MCAR** — missing completely
at random — meaning the mask `M` (which entries are present) is statistically independent of the
data `X`. (The weaker assumptions, MAR — missingness depends only on observed values — and MNAR —
missingness depends on the unobserved values themselves — are the realistic generalizations, but
the clean guarantees are proved under MCAR.)

## Background

**Notation and types of missingness.** Write `X = (X_1, …, X_d)` for the data vector with
distribution `P(X)`, and `M = (M_1, …, M_d) ∈ {0,1}^d` for the *mask*, with `M_i = 1` meaning
component `i` is observed. The partially observed vector is `X̃`, where `X̃_i = X_i` if `M_i = 1` and
`X̃_i = *` (a special "unobserved" symbol) otherwise; the mask is recoverable from `X̃`. Missingness
is **MCAR** if `M ⟂ X`; **MAR** if the missingness depends only on observed entries; **MNAR**
otherwise. The dataset is `{(x̃^i, m^i)}_{i=1}^n`. A central modeling choice is whether to target the
conditional *mean* of the missing entries or their full conditional *distribution* `P(X | X̃)`; the
latter, of dimension equal to the number of missing coordinates `‖1 − M‖_1`, supports multiple
imputation (drawing several distinct completions) and uncertainty quantification.

**The generative vs. discriminative split.** Imputation methods at this time fall into two camps.
*Discriminative* methods directly predict each missing entry from the others (chained regressions,
random forests, low-rank matrix completion). *Generative* methods posit a model of the data
distribution and sample missing values from it (Expectation-Maximization on a parametric family,
and the newer deep-learning approaches). Each camp has a characteristic weakness that a better
method would need to overcome (detailed under Baselines).

**Adversarial training of generative models (Goodfellow et al. 2014).** A pair of networks is
trained against each other: a generator `G` maps a noise prior `p_z(z)` into data space, inducing a
distribution `p_g`; a discriminator `D` outputs the probability that a sample came from the real
data rather than from `G`. They play the two-player minimax game

```
min_G max_D  V(D, G) = E_{x ~ p_data}[ log D(x) ] + E_{z ~ p_z}[ log(1 − D(G(z))) ].
```

The load-bearing facts of this framework, which any adaptation rests on:
- *Optimal discriminator.* For a fixed `G`, the best `D` is `D*(x) = p_data(x) / (p_data(x) + p_g(x))`.
  This follows from a single pointwise fact: the map `y ↦ a log y + b log(1 − y)` is maximized on
  `[0, 1]` at `y = a/(a + b)` for `a, b ≥ 0` not both zero.
- *What the generator then optimizes.* Substituting `D*` back, the generator's criterion becomes
  `C(G) = −log 4 + 2·JSD(p_data ‖ p_g)`, where `JSD` is the Jensen–Shannon divergence. Because
  `JSD ≥ 0` with equality iff the two distributions coincide, the global minimum is achieved exactly
  when `p_g = p_data`. The adversarial game, at its optimum, recovers the true data distribution.
- *Practice.* The game is solved iteratively: alternate `k` stochastic-gradient steps on `D` with
  one on `G`, on minibatches; `G` and `D` are multilayer perceptrons. Early in training
  `log(1 − D(G(z)))` saturates because `D` rejects obvious fakes with high confidence, giving `G`
  vanishing gradient; the standard remedy is to instead have `G` *maximize* `log D(G(z))`, which has
  the same fixed point but much stronger early gradients.

**Conditioning a generative model (Mirza & Osindero 2014).** Both `G` and `D` can be conditioned on
auxiliary information `y` (a label, another modality) by feeding `y` as an extra input to each
network; the game becomes `min_G max_D E[ log D(x | y) ] + E[ log(1 − D(G(z | y))) ]`. This turns an
unconditional sampler into a conditional one — a one-to-many mapping from the conditioning variable
to a distribution over outputs.

**Autoencoders for reconstruction.** A denoising autoencoder (Vincent et al. 2008) corrupts its
input, encodes it to a bottleneck, and decodes back, training the reconstruction to match the clean
input; it learns to fill in / denoise corrupted entries. This reconstruction idea — force the
network's output on the *observed* coordinates to match what was actually observed — is a natural
supervisory signal whenever part of the input is known.

**Representational limits in the design space.** Two observations frame the problem. (1)
Mean/median filling, which replaces a column's holes by that column's average, is fast
but discards every cross-feature dependency, so it systematically distorts correlations and the
joint distribution. (2) Methods that learn a *point* predictor of each missing entry recover (at
best) the conditional mean and therefore cannot express the spread of plausible values — they
cannot do multiple imputation.

## Baselines

These are the prior imputation methods commonly used as comparators for this task.

**Mean (column-mean) imputation.** Replace each missing entry in a column by the mean of that
column's observed entries (estimated on training data). *Idea/math:* `x̂_{ij} = mean_i` for missing
`(i, j)`. **Gap:** ignores all feature correlations; every record gets the same fill for a given
column, collapsing the conditional distribution to a constant and biasing downstream covariances.

**k-Nearest-Neighbors imputation (Troyanskaya et al., Bioinformatics 2001).** For each record with a
missing entry, find the `k` most similar records using only the features both records have observed,
and impute the missing entry as the (distance-weighted) average of those neighbors' values; default
`k = 5`. *Idea/math:* a local, nonparametric conditional-mean estimate. **Gap:** distance computed
on observed features degrades as missingness grows; it still produces an averaged point estimate, so
no uncertainty; and it scales poorly (a neighbor search per query) on large datasets.

**MICE — Multivariate Imputation by Chained Equations (van Buuren & Groothuis-Oudshoorn, JSS 2011).**
Initialize the holes (e.g. by column means), then cycle: for each variable with missingness, fit a
regression of that variable on all the others using the current filled-in values, and replace its
missing entries with draws from the fitted predictive model; repeat for several rounds (default
`max_iter = 10`). *Idea/math:* a Gibbs-like sweep over per-variable conditional models; sampling
from each conditional (rather than taking the mean) lets MICE produce multiple imputations.
`sklearn.impute.IterativeImputer` is the de facto implementation. **Gap:** the conditionals are
modeled separately and (typically) linearly, so the implied joint may be incoherent and complex
nonlinear interactions are missed; quality hinges on the choice of per-variable model.

**MissForest (Stekhoven & Bühlmann, Bioinformatics 2012).** The same chained-equations skeleton as
MICE, but each per-variable predictor is a Random Forest. *Idea/math:* iteratively regress each
variable on the rest with a forest, iterating until the change in imputations stops decreasing.
**Gap:** captures nonlinearities and mixed types well, but it is a *discriminative* point predictor
fit per variable — it returns the forest's conditional-mean estimate, so it does not model the joint
distribution and cannot natively express imputation uncertainty; cost grows with the number of
features (a forest per feature per iteration).

**Low-rank matrix completion (Mazumder, Hastie & Tibshirani 2010).** Treat the data table as a
partially observed matrix and recover it by minimizing a nuclear-norm-regularized fit to the
observed entries (soft-thresholded SVD). *Idea/math:* assumes the complete matrix is approximately
low rank. **Gap:** the low-rank/linear-subspace assumption is restrictive; it does not model a
distribution and struggles with strongly nonlinear feature relationships and mixed variable types.

**Expectation-Maximization on a parametric family (e.g. a joint Gaussian).** Fit a parametric joint
distribution to the incomplete data by EM, then impute missing entries from the fitted conditionals.
*Idea/math:* alternate computing expected sufficient statistics given current parameters (E) with
re-estimating parameters (M). **Gap:** commits to a parametric form for `P(X)` and degrades badly
when that form is wrong — in particular for datasets mixing categorical and continuous variables.

**Deep autoencoder imputation (denoising autoencoders; Gondara & Wang 2017; Vincent et al. 2008).**
Train an autoencoder to reconstruct records, using its decoder output on the missing coordinates as
the imputation. *Idea/math:* learn a nonlinear encode-decode of the data manifold and read off the
reconstructed holes. **Gap:** the classic denoising-autoencoder recipe needs *complete* data during
training (it corrupts clean inputs to create the (corrupted, clean) pairs it learns from), which is
exactly what is unavailable when missingness is intrinsic; variants that train on incomplete data
only use the observed components and learn no model of the full joint distribution.

The shape of the open problem these leave: the discriminative methods (kNN, MICE, MissForest, matrix
completion) are point predictors that recover at most a conditional mean and so cannot represent
imputation uncertainty; the generative ones either commit to a parametric family that fails on mixed
data (EM) or require complete training data (denoising autoencoders). None offers a way to learn a
flexible, nonparametric model of the full conditional distribution `P(X | X̃)` directly from data
that is itself incomplete.

## Evaluation settings

Natural yardsticks for the task:

- **Datasets.** Tabular datasets with a mix of continuous and binary/categorical features. UCI
  benchmarks in common use for imputation: Breast Cancer (569 samples, 30 continuous features),
  Spam (4,601 × 57), Letter (20,000 × 16 categorical), Credit (30,000, mixed), News (≈40,000,
  mixed). Smaller standard classification sets are also used: Breast Cancer Wisconsin (569 × 30,
  binary label), Wine (178 × 13, 3 classes), and the California Housing regression set
  (≈5,000 × 8). Each carries its own joint structure and average inter-feature correlation, which
  governs how much an imputer can exploit.
- **Corruption protocol.** Introduce missingness artificially so the ground truth is known for
  scoring: remove a fixed fraction of all entries uniformly at random — **MCAR at a 20% missing
  rate** is the standard setting — optionally sweeping the rate to study robustness. (MAR and MNAR
  corruption schemes exist as harder variants.)
- **Metrics.** *Imputation accuracy* — RMSE between imputed and true values on the held-out masked
  entries (lower is better); for categorical features the proportion of falsely classified entries.
  *Downstream utility* — train a fixed predictive model (e.g. gradient boosting, or logistic
  regression) on the imputed data and measure its test performance (accuracy / AUROC for
  classification, R² for regression); a good imputer should preserve the feature-label relationship.
- **Protocol.** Repeat each experiment many times with cross-validation and report mean ± std,
  comparing all methods under the identical missingness mask and the same downstream predictor for
  fairness.

## Code framework

A standard imputer receives a data matrix containing `NaN`s and returns a finite same-shape matrix,
without using test labels. The available substrate is generic: numpy/scipy, min-max scaling, an MLP
layer with a nonlinearity, gradient-based optimizers, and standard elementwise losses such as
squared error and binary cross-entropy. The learning rule remains an open slot.

```python
import numpy as np


def minmax_normalize(data, parameters=None):
    """Scale each column to [0, 1] using observed entries."""
    if parameters is None:
        mn = np.nanmin(data, axis=0)
        scale = np.nanmax(data, axis=0) - mn
        return (data - mn) / (scale + 1e-6), {"min": mn, "scale": scale}
    return (data - parameters["min"]) / (parameters["scale"] + 1e-6), parameters


def minmax_denormalize(norm_data, parameters):
    return norm_data * (parameters["scale"] + 1e-6) + parameters["min"]


def mlp_layer(inp, W, b, activation):
    """A single fully-connected layer; activation is one of the standard nonlinearities."""
    z = inp @ W + b
    return activation(z)


def bce(target, pred, eps=1e-8):
    """Elementwise binary cross-entropy (already-available loss primitive)."""
    return -(target * np.log(pred + eps) + (1 - target) * np.log(1 - pred + eps))


def squared_error(target, pred):
    return (target - pred) ** 2


def round_near_categorical(imputed, original):
    rounded = imputed.copy()
    for j in range(original.shape[1]):
        observed = original[~np.isnan(original[:, j]), j]
        if len(np.unique(observed)) < 20:
            rounded[:, j] = np.round(rounded[:, j])
    return rounded


def learn_filler(norm_data, mask, parameters):
    # TODO: learn the missing-value filler from the incomplete normalized data.
    raise NotImplementedError


def impute_missing(data_x, parameters=None):
    parameters = {} if parameters is None else parameters
    mask = 1.0 - np.isnan(data_x).astype(float)
    norm_data, norm_parameters = minmax_normalize(data_x)
    norm_data = np.nan_to_num(norm_data, nan=0.0)
    filled_norm = learn_filler(norm_data, mask, parameters)
    filled_norm = mask * norm_data + (1.0 - mask) * filled_norm
    filled = minmax_denormalize(filled_norm, norm_parameters)
    return round_near_categorical(filled, data_x)
```

The mask is read off the data as `mask = 1 - np.isnan(data_x)`. The only missing piece is the
learning rule that maps the normalized incomplete matrix and its mask to a normalized completion.
