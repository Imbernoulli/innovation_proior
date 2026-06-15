# Context: subgroup-reliable post-hoc calibration of a fixed classifier

## Research question

A binary classifier is already trained and frozen; it emits a positive-class probability
`p` for each example. We want those probabilities to be *calibrated* — among the examples
the model calls "70% positive", about 70% really are positive — and we want this to hold not
only on average but **within each subgroup** (defined by a protected or operationally
meaningful attribute), and to **survive a shift** between the data we calibrate on and the
data we are scored on. Calibration is a post-processing step: we may only learn a mapping
from the model's raw probability `p` (optionally using the example's subgroup id `g`) to a
corrected probability `q`, with the base model and the data splits fixed.

Two things make this hard at once. First, average calibration is cheap but misleading: a map
can look perfect pooled over everyone while being badly over-confident on one subgroup and
under-confident on another, and any decision made by thresholding the confidence is then
systematically biased for some groups. So the quantity that actually matters is the *worst*
subgroup's calibration error, not the mean. Second, each subgroup's calibration sample is
small — a few hundred points, sometimes a few dozen — and the test distribution is
deliberately different from the calibration distribution (a domain score selects a held-out
tail). A calibrator with much capacity will fit the idiosyncrasies of a small group's
calibration sample and then generalize poorly to that group's shifted test points. A solution
has to improve the worst subgroup without paying for it in variance on the small subgroups.

## Background

**Why a trained classifier is miscalibrated in the first place.** Modern high-capacity
classifiers trained to minimize log-loss are systematically *over-confident*: once almost
every training point is classified correctly, the log-loss can still be driven down by
pushing the predicted probabilities further toward 0 and 1, so the model keeps sharpening its
outputs past the point where they reflect real frequencies. This is a diagnostic fact about
trained models, observed before any correction is applied: log-loss keeps overfitting while
the 0/1 error has stopped improving (and can even keep improving slightly), so the excess is
spent entirely on probability mass, i.e. on confidence. The dominant, simplest-to-describe
form of this miscalibration is that the *scale* of the score is wrong — the model's logits
are uniformly too large — rather than the ordering being wrong.

**The logit representation.** For a binary model the positive-class probability and the
underlying real-valued score are related by the logistic link
`p = sigma(z) = 1/(1+e^{-z})`, equivalently `z = logit(p) = log(p/(1-p))`. The logit `z` is the
natural unconstrained coordinate: `sigma` and `logit` are the canonical monotone bijection
between `[0,1]` and `R`, so any monotone transformation of `p` can be expressed as a
transformation of `z`, and a transformation of `z` that is monotone increasing leaves the
example ranking — and the `p = 0.5 <-> z = 0` decision boundary — untouched. Acting on logits
is therefore the way to soften or sharpen confidences *without* changing which class is
predicted, and hence without changing accuracy.

**Proper scoring rules vs. the calibration metric.** The objective used to *fit* a calibrator
is the negative log-likelihood (binary cross-entropy),
`NLL = -mean( y log q + (1-y) log(1-q) )`. NLL is a strictly proper scoring rule: in
expectation it is minimized exactly when the reported probability equals the true conditional
probability, so descending it pushes `q` toward calibration. The metric used to *evaluate*
calibration is the Expected Calibration Error, computed by binning predictions into `M`
equal-width confidence bins `B_m` and averaging the gap between accuracy and mean confidence,
`ECE = sum_m (|B_m|/n) |acc(B_m) - conf(B_m)|`. ECE is non-differentiable (it bins), so it is
a yardstick, not a training loss; the maximum-over-groups of ECE is the worst-group target.

**Shrinkage / partial pooling — the statistics of estimating many related quantities.** When
many related quantities must be estimated from samples of very different sizes, estimating each
quantity in isolation by its own sample is often wasteful. Stein (1956) and James & Stein
(1961) showed that for `X ~ N(theta, sigma^2 I)` with `p >= 3` coordinates, the obvious
estimator "report each coordinate's own observation" is *inadmissible*: an estimator that pulls
the noisy coordinates toward a common center has lower total squared-error risk. Efron & Morris
(1973, 1975) recast the same phenomenon as empirical Bayes: a noisy local estimate can be
combined with a grand mean, with the amount of pooling governed by the relative precision of
the local sample and the spread of true values across groups. The useful lesson is a
statistical warning: when a task asks for many related estimates and some of them are based on
small samples, independent fitting is exactly where variance can overwhelm the signal.

## Baselines

These are the post-hoc calibrators a subgroup-reliable method would be measured against.

**Global temperature scaling (Guo, Pleiss, Sun & Weinberger, ICML 2017; binary form of Platt
scaling, Platt 1999).** Divide the logit by a single positive scalar `T` before the link:
`q = sigma(z/T)`. `T` is fit by minimizing validation NLL. `T = 1` is the identity, `T > 1`
softens (raises entropy, fixes over-confidence), `T -> infinity` sends `q -> 1/2`, `T -> 0`
sends it to a hard 0/1. Because `T > 0` is a single shared scalar it is monotone in `z`, so it
preserves ranking and the decision threshold — accuracy is untouched — and it can be read as
the maximum-entropy correction subject to matching the average true-class logit to the average
expected logit, i.e. it precisely targets the "logits are uniformly too large" failure mode.
**Limitation:** one scalar for the whole population. If different subgroups need different
amounts of softening — because their score distributions or prevalences differ — a single `T`
cannot satisfy them simultaneously; it minimizes pooled NLL and can leave some subgroups
over-confident and others under-confident, so the worst-group calibration error stays high.

**Platt scaling (Platt 1999).** The two-parameter logistic fit `q = sigma(a z + b)`, with
`a, b` fit by NLL on held-out scores of a frozen model. The slope `a` rescales and the
intercept `b` shifts the logits. Temperature scaling is its one-parameter special case
(`a = 1/T`, `b = 0`). **Limitation:** the extra intercept can shift the decision boundary
(`a z + b = 0` is not `z = 0`), so it can change predictions; and with two parameters it has
more capacity to fit a small calibration sample than a single scalar does.

**Isotonic regression (Zadrozny & Elkan 2002).** Fit the best non-decreasing step function
`q = f(p)` by least squares with `f` monotone (pool-adjacent-violators). Non-parametric, so it
can represent essentially any monotone miscalibration shape. **Limitation:** its effective
number of parameters grows with the data, so on a small sample it overfits and produces coarse
piecewise-constant probabilities; fit per subgroup on a few dozen points it is very high
variance.

**Beta calibration (Kull, Silva Filho & Flach 2017).** A three-parameter family
`q = sigma(c + a log p - b log(1-p))` derived from assuming class-conditional Beta densities
on the scores; it subsumes the identity, sigmoids and inverse sigmoids and can correct more
than a pure scale error. **Limitation:** three parameters is still more capacity than a single
scalar, so per small subgroup it is more prone to overfit the calibration sample and shift the
threshold than temperature scaling.

The common thread: the single global scalar is robust but cannot adapt to subgroups; every
more-flexible calibrator can adapt but, fit independently on a small per-subgroup sample,
becomes high-variance and degrades under test-time shift.

## Evaluation settings

Cached high-stakes tabular datasets (from AIF360) with protected-attribute subgroups:

- **Adult** (Census income): predict income `> 50k`; subgroups from `sex`, `race`.
- **COMPAS** (recidivism risk): subgroups from `race`, `sex`.
- **Law School GPA**: outcome binarized at the median first-year GPA; subgroups from `race`,
  `gender`.

For each dataset the base classifier is a standardized logistic-regression pipeline
(`class_weight="balanced"`). The split is intentionally shifted: per class, examples are
ordered by a domain score (e.g. `age`/`hours-per-week`; `priors_count`/`age`; `lsat`/`ugpa`),
the top tail is held out as test, calibration is a random subset of the remaining source
region, and the rest is training. Subgroups are the cross-product of the protected attributes.
Metrics: **`worst_group_ece`** (worst-subgroup ECE, 15 equal-width bins; primary, lower
better), **`brier`** (test Brier score, lower better), **`max_subgroup_gap`** (max minus min
subgroup ECE, lower better), and **`subgroup_auroc`** (mean per-subgroup AUROC, higher better,
diagnostic). Subgroups with fewer than 5 test points are skipped in the metric.

## Code framework

The calibrator plugs into a fixed harness: a logistic base classifier is trained on the
training split, its positive-class probabilities are read off on the calibration and test
splits, and a `CalibrationMethod` is `fit` on calibration `(probs, labels, groups)` and then
applied with `predict_proba` to both. The numerical primitives already exist — `numpy`,
`scipy.special.logit`/`expit` for the logit link, `scipy.optimize.minimize_scalar` for a
one-dimensional fit, the binned-ECE and Brier functions for evaluation. What does **not** yet
exist is the mapping itself: how the raw probabilities (and the optional subgroup ids) become
calibrated probabilities. That is the single empty slot.

```python
import numpy as np
from scipy import optimize, special


class CalibrationMethod:
    """Map raw positive-class probabilities to calibrated ones.

    fit() sees calibration (probs, labels) and optional subgroup ids;
    predict_proba() returns calibrated probabilities in [eps, 1 - eps].
    groups may be None (a group-agnostic calibrator ignores it).
    """

    def __init__(self):
        self.eps = 1e-6
        # TODO: any state the calibration mapping we design will need.

    def fit(self, probs, labels, groups=None):
        probs = np.asarray(probs, dtype=float).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)
        # TODO: learn the calibration mapping from (probs, labels, groups).
        return self

    def predict_proba(self, probs, groups=None):
        probs = np.asarray(probs, dtype=float).reshape(-1)
        # TODO: apply the learned mapping; return valid probabilities.
        return np.clip(probs, self.eps, 1.0 - self.eps)


# fixed harness (already provided)
def run(base_model, X_calib, y_calib, g_calib, X_test, g_test):
    cal_probs = base_model.predict_proba(X_calib)[:, 1]   # raw positive-class probs
    test_probs = base_model.predict_proba(X_test)[:, 1]
    method = CalibrationMethod().fit(cal_probs, y_calib, groups=g_calib)
    return method.predict_proba(test_probs, groups=g_test)  # calibrated test probs
```
