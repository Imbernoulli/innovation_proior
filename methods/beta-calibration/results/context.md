# Context: post-hoc probability calibration of binary classifiers (circa 2016)

## Research question

Many classifiers output a score `s = f(x)` in `[0, 1]` that is treated as a probability of
the positive class but is not one. The aim of *calibration* is to apply a map `mu` on top of
the score so that `mu(f(x))` is a trustworthy posterior probability — one that can be
thresholded directly for cost-sensitive or prior-shifted decisions without re-optimisation.
Formally, a scoring classifier is perfectly calibrated on a dataset when, for each output
value `s_i = f(x_i)`,

```
s_i = E[Y | f(X) = s_i],
```

i.e. among instances that receive score `s_i`, the proportion of positives is `s_i`. There is
a unique map that achieves this on a given dataset, `mu(s_i) = E[Y | f(X) = s_i]`, but fitting
it directly overfits: if `f` outputs a distinct score on every training instance, the
training-perfect map collapses to the 0/1 labels. So calibration is done with a *parametric
family* carrying an inductive bias, fitted on a held-out calibration set.

The question here is what parametric calibration family to use: a family of maps from the raw
score to a calibrated posterior, fitted cheaply on a small held-out calibration set, that can
undo the kinds of score distortion real classifiers produce.

## Background

By this time post-hoc calibration is a routine step between a scoring model and a decision
rule. The motivation is decision-theoretic: the closer `mu(f(x))` is to the true class
posterior, the closer thresholding gets to the irreducible Bayes risk, and a calibrated score
can be re-thresholded analytically when misclassification costs or the class prior change. The
field state and the load-bearing facts:

- **Score distortions are systematic and classifier-dependent, and they have been measured.**
  Reliability diagrams (binned score against empirical positive rate) reveal that different
  model families distort their scores in characteristic, repeatable ways. Maximum-margin
  methods and boosting push scores *toward the middle*, producing a sigmoid-shaped reliability
  curve — the empirical positive rate is more extreme than the score. This is the case the
  standard parametric method was built for. Other widely used classifiers do the opposite.
  Naive Bayes multiplies many per-feature likelihood ratios and, when features are correlated,
  double-counts evidence, so its scores are pushed *to the extremes* near 0 and 1; the
  original-formulation Adaboost (probabilities read off the additive-logistic-regression view
  of Friedman, Hastie & Tibshirani 2000) does the same. For those, the empirical positive rate
  is *less* extreme than the score, an inverse-sigmoid-shaped reliability curve.

- **A score that lives on `[0, 1]` is a bounded quantity.** A model of the per-class score
  distribution that places mass outside `[0, 1]` is incoherent for such scores.

- **The likelihood-ratio route to a calibration map.** A way to derive a parametric
  calibration family is to posit a density for the score *within each class*, `p(s | +)` and
  `p(s | -)`, form the likelihood ratio `LR(s) = p(s | +) / p(s | -)`, and (under a uniform
  class prior, so the likelihood ratio equals the posterior odds) read off the calibrated
  posterior `mu(s) = 1 / (1 + LR(s)^{-1})`. The shape of the resulting map is determined by the
  assumed per-class densities. This is the device that produced the standard sigmoid family.

- **Fitting by log-loss is fitting by maximum likelihood.** The standard goodness-of-fit
  measure for probability estimates is log-loss,
  `LL(p_hat, y) = sum_i [ -y_i ln p_hat_i - (1 - y_i) ln(1 - p_hat_i) ]`, which equals the
  negative log-likelihood of the labels under the predicted probabilities. Minimising log-loss
  is maximum-likelihood estimation. So a calibration family whose map can be written as a
  logistic-regression posterior in some transformed feature(s) can be fitted by an existing
  logistic-regression solver, rather than by a bespoke optimiser.

## Baselines

**Logistic (Platt) calibration (Platt, 1999; for SVMs).** Fit a sigmoid to the score,
`mu_logistic(s; gamma, delta) = 1 / (1 + 1/exp(gamma * s + delta))`, with `gamma >= 0` so the
map is monotonically non-decreasing. Reparametrising with the midpoint `m = -delta/gamma` gives
`mu_logistic(s; gamma, -m*gamma) = 1 / (1 + 1/exp(gamma*(s - m)))`: `m` is the score at which
the calibrated value is `1/2`, and the slope there is `gamma/4`. The family has a first-principles
justification: assume the per-class scores are *normally distributed with equal variance*
`sigma^2`, means `s_+` and `s_-`. Then

```
LR(s) = p(s|+)/p(s|-)
      = exp[ ( -(s - s_+)^2 + (s - s_-)^2 ) / (2 sigma^2) ]
      = exp[ (s_+ - s_-)/sigma^2 * ( s - (s_+ + s_-)/2 ) ]
      = exp[ gamma (s - m) ],   gamma = (s_+ - s_-)/sigma^2,  m = (s_+ + s_-)/2,
```

and `mu = 1/(1 + LR(s)^{-1}) = 1/(1 + exp(-gamma(s - m)))`, exactly the sigmoid. Conversely
every such sigmoid corresponds to a pair of equal-variance Gaussians. Platt also
considered the richer two-Gaussian-*unequal*-variance posterior `1/(1 + exp(a f^2 + b f + c))`
and rejected it because it is non-monotonic in the score. Fitting is maximum-likelihood on the
cross-entropy, i.e. univariate logistic regression with feature `s` and label `y`, fit on a
held-out / cross-validated set.

**Isotonic calibration (Zadrozny & Elkan, 2002; pair-adjacent-violators, Ayer et al., 1955).**
Non-parametric: fit the best monotonically non-decreasing step function from scores to labels,
`min_{m isotonic} sum_i (y_i - m(f(x_i)))^2`, solved by the pair-adjacent-violators algorithm
(equivalently, the slopes of the ROC convex hull read as empirical likelihood ratios). It can
capture any monotonic distortion. Learning-curve analysis (Niculescu-Mizil & Caruana, 2005)
characterises how much calibration data it needs relative to the parametric sigmoid; its output
is piecewise-constant.

**Linear-in-log-odds recalibration (Lichtenstein, Fischhoff & Phillips, 1977; Turner et al.,
2014).** Map the log-odds of the score linearly and squash back through a sigmoid,
`mu(s) = 1/(1 + 1/exp(a * ln(s/(1-s)) + c))`. Used in the human-forecasting and
forecast-aggregation literature to sharpen or flatten probabilities. It is offered as an
empirical transform with a single shape parameter on the log-odds.

## Evaluation settings

The natural yardsticks already in use for calibration:

- A bank of binary classification datasets — in this regime, UCI datasets, with
  multiclass datasets binarised by calling the largest class positive and the rest negative —
  spanning a wide range of sample sizes so that small-calibration-set behaviour is exercised.
- Base classifiers whose score distortions are known to differ: Naive Bayes and Adaboost in
  two formulations (original-style probabilities from the additive-logistic-regression view,
  and the SAMME variant), since the former two push scores to the extremes and the latter pulls
  them toward the middle — covering both distortion directions.
- A nested resampling protocol: repeated stratified cross-validation, with an inner split that
  holds out a fold purely to fit the calibration map (the model is trained on the other folds),
  averaging the resulting calibrated classifiers — the protocol used for the standard sigmoid
  method.
- Metrics: log-loss and Brier score, both proper scoring rules and therefore well-founded
  measures of probability quality. Statistical comparison across datasets by the Friedman test
  with a Nemenyi post-hoc and critical-difference diagrams.

## Code framework

A calibration map plugs in between a fixed base classifier and the metric computation: the base
model produces positive-class scores on a held-out calibration split (with labels) and on a
test split; the calibration object is fitted on the former and applied to the latter. The
substrate that already exists is the score/label arrays, a numerically-safe clip to keep scores
strictly inside `(0, 1)`, and an off-the-shelf logistic-regression solver. The calibration
family itself — the per-class score model, the resulting map, and the feature transform (if
any) that would let an existing solver fit it — is the empty slot.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression


class CalibrationMethod:
    """Maps raw positive-class scores in (0, 1) to calibrated positive-class
    probabilities. Fitted on a held-out calibration split, applied to test."""

    def __init__(self):
        self.eps = 1e-6
        # TODO: any parametric state the calibration map we design will need.

    def _featurize(self, probs):
        probs = np.asarray(probs).reshape(-1)
        p = np.clip(probs, self.eps, 1.0 - self.eps)   # keep scores inside (0, 1)
        # TODO: the transform of the score we will design, into whatever space
        #       makes the calibration map fittable by a standard solver.
        raise NotImplementedError

    def fit(self, probs, labels, groups=None):
        labels = np.asarray(labels).reshape(-1).astype(int)
        # TODO: fit the calibration map on (score, label) pairs.
        raise NotImplementedError

    def predict_proba(self, probs, groups=None):
        # TODO: apply the fitted calibration map; return calibrated probs in [0, 1].
        raise NotImplementedError
```

The outer harness supplies calibration scores+labels to `fit` and test scores to
`predict_proba`; the design of the map and its feature transform is what fills the stubs.
</content>
</invoke>
