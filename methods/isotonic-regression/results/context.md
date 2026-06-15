# Context: turning classifier scores into calibrated probabilities (circa 2001-2002)

## Research question

A trained classifier hands us, for each example `x`, a real-valued score `s(x)` — a naive-Bayes
posterior, an SVM's signed distance to the separating hyperplane, a boosted model's margin. For
*ranking*, these scores are often excellent: if `s(x) > s(y)` then `x` really is more likely than `y`
to be in the class. But ranking is not enough for a large and growing set of uses. When a
classification output is *combined with other information* to make a decision — multiplied by an
example-dependent misclassification cost, fed as one input to a higher-level model (a language model
over character-recognizer outputs, say), or merged with the outputs of other classifiers — what is
needed is the actual *probability* of class membership, on an absolute scale, not merely a position in
an ordering. A classifier is **well-calibrated** when, among all examples it scores near `p`, the
empirical fraction that truly belong to the class is near `p`: of the examples assigned score `0.8`,
about 80% should be positives. Modern classifiers routinely fail this test even when they rank well.

The precise problem: the base classifier and the train / calibration / test splits are fixed. From a
held-out labeled calibration set, learn a **post-hoc mapping** `s(x) ↦ p̂(x)` that takes the
classifier's raw, possibly distorted scores and returns numbers that *are* calibrated probabilities —
non-negative, and (for the multiclass case, over all classes) summing to one — without retraining the
base model and without sacrificing the ranking that the scores already get right. A solution has to fix
whatever shape of distortion the classifier happens to have, regularize enough to generalize from a
finite calibration set, and remain cheap to fit and to apply.

## Background

The notion of calibration comes from the forecasting literature (Murphy & Winkler 1977; DeGroot &
Fienberg 1982). A score function `s` is well-calibrated if the empirical class-membership probability
`P(c | s(x) = s)` converges to the value `s` as the sample grows. The standard diagnostic is the
**reliability diagram**: discretize the score range into bins, and for each bin plot the bin's score
against the empirical positive rate `P(c | s(x) ≈ s)` in that bin. A perfectly calibrated classifier's
points all land on the `y = x` diagonal; deviations off the diagonal are exactly the miscalibration to
be corrected.

Several already-visible calibration failures motivate the problem:

- **Naive Bayes scores are systematically too extreme.** Naive Bayes multiplies per-attribute
  likelihoods as if attributes were conditionally independent; real attributes are correlated, so the
  product over-counts evidence and pushes posteriors toward 0 or 1. On a reliability diagram its points
  bow away from the diagonal — but larger scores still generally correspond to larger empirical
  probabilities. The ranking survives; only the values are wrong.
- **SVM "scores" are not probabilities at all.** An SVM outputs `f(x)`, the signed distance of `x` from
  the separating hyperplane, ranging over `[-a, a]`. Magnitude is a confidence proxy (points far from
  the boundary are classified more confidently), but distance is not proportional to probability.
  Re-scaling into `[0, 1]` by `s(x) = f(x)/(2a) + 1/2` makes
  `f = 0` map to `0.5`, but the re-scaled scores are still uncalibrated, and on several datasets the
  empirical-probability-versus-score curve has a pronounced **sigmoidal** shape.
- For both, the picture is the same: scores **rank** examples correctly while reporting the **wrong
  numbers**. There is useful signal in the ordering, and the task is to recover the correct vertical
  scale.

A practical obstacle underlies everything: in real data the number of distinct score values is large
relative to the number of labeled examples, so we cannot just read `P(c | s(x) = s)` off the data for
every `s` — there are too few examples at each exact score to estimate a reliable empirical
probability. Any method has to *aggregate* nearby scores to get stable estimates, and the question of
*how* to aggregate is where the methods differ. A bias-variance tension runs through it: aggregate too
coarsely (or impose too rigid a shape) and you cannot represent the true distortion; aggregate too
finely (or impose nothing) and the estimates are noisy and overfit the calibration set.

A second, separate difficulty is the **multiclass** case. Calibration as defined above is per-class and
transfers directly to `k` classes — class `c_i` is calibrated if `P(c_i | s(c_i, x) = s) → s`. But a
classifier's full output is a point in the `(k-1)`-dimensional probability simplex, and learning a
correction *in that joint space* is hard: there is no obvious "shape" to impose on a map from a
`(k-1)`-dimensional space to itself, and non-parametric estimates degrade as `k` grows (curse of
dimensionality). The one-dimensional structure that makes the binary problem tractable is absent. Tools
do exist for *combining* binary classifiers into multiclass predictions — Allwein, Schapire & Singer
(2000) show any reduction can be written as a code matrix `M ∈ {-1,0,1}^{k×l}`, generalizing
error-correcting output codes (Dietterich & Bakiri 1995); least-squares with non-negativity (Kong &
Dietterich 1997) and log-loss "coupling" (Hastie & Tibshirani 1998; Zadrozny 2002) recover a
distribution from the per-column estimates — but these are reduction/combination machinery, not
calibration of the per-class probabilities themselves.

## Baselines

**Sigmoid / Platt scaling (Platt 1999; for naive Bayes, Bennett 2000).** Fit a two-parameter sigmoid
mapping the raw score to a probability,

```
p̂(c | x) = 1 / (1 + exp(A · s(x) + B)),
```

choosing `A, B` to minimize the negative log-likelihood of the calibration labels (with a small
regularization of the target labels to avoid overfitting). Motivated by the observation that the
score-to-probability curve for SVM margins is sigmoidal on many datasets; Platt showed it gives
probabilities at least as accurate as training an SVM specifically for probabilities, while being
faster. With only two free parameters it is strongly regularized and works well on small calibration
sets. **Gap:** it *commits to a single functional shape*. When the true distortion really is sigmoidal
(SVM on Adult, say) it fits beautifully, but when the score-to-probability relationship has some other
increasing shape — as naive-Bayes curves can — a sigmoid cannot represent it, and the residual
miscalibration is whatever the sigmoid leaves behind. It can only undo distortions of the one shape it
assumes.

**Binning / histogram estimation (Zadrozny & Elkan 2001).** A shape-free, non-parametric alternative.
Sort the calibration examples by score and split the sorted list into `b` subsets of equal size — the
bins. Record each bin's lower/upper score boundaries. To score a test example, find the bin its score
falls in and return that bin's *fraction of positive training examples* as the calibrated probability.
**Gaps:** the number of bins `b` must be chosen by cross-validation, which is unreliable when the
calibration set is small or highly unbalanced; and, more fundamentally, the **bin size is fixed and the
boundary positions are essentially arbitrary** — they are dictated by equal counts, not by where the
score actually carries information. If a boundary happens to fall in the middle of a region where the
true probability is changing fast, the bin averages together examples that clearly deserve different
probabilities and smears them into one value. Coarse where it should be fine, and fine where coarse
would do; the granularity is not adapted to how well the classifier ranks in each part of the range.

**Directly calibrating the multiclass joint output.** One could try to learn the correction in the
`(k-1)`-dimensional simplex directly. **Gap:** there is no sensible shape constraint to impose on a
high-dimensional simplex-to-simplex map, and non-parametric estimation in that many dimensions needs
prohibitively much data as the number of classes grows; the estimates are not reliable.

**Code-matrix combiners for multiclass (Kong & Dietterich 1997; Hastie & Tibshirani 1998;
Zadrozny 2002).** Given per-binary-problem probability estimates `r_b(x)` and a code matrix `M`, recover
a class distribution `P(c | x)` either by least-squares with non-negativity constraints or by an
iterative log-loss "coupling" procedure. These solve the *combination* step for arbitrary code matrices.
**Gap:** they are general-purpose reduction machinery and bring more apparatus than the simplest
reduction needs; they say nothing about how to calibrate the per-class scores that feed them.

## Evaluation settings

The yardsticks already in use for probability quality:

- **Reliability diagrams** for qualitative calibration assessment: score bins on the x-axis, empirical
  positive rate on the y-axis, compared to the diagonal.
- **Mean squared error / Brier score** (Brier 1950) as the primary quantitative metric. For one example,
  the squared error is `Σ_c (T(c,x) − P̂(c,x))²`, where `T(c,x) = 1` if the true label of `x` is `c` and
  `0` otherwise; average over examples for MSE on a set. DeGroot & Fienberg (1982) decompose MSE into a
  calibration component and a refinement component, so a well-calibrated, more refined (more confident)
  classifier scores lower.
- **Negative log-likelihood / cross-entropy** between predicted distribution and the one-hot label.
- **Error rate**, to check the correction does not damage the underlying classification.
- Datasets and base learners that exist at the time: two-class direct-marketing data (KDD-98 charity
  donations, with a profit metric; The Insurance Company / COIL-2000), the Adult income dataset, text
  categorization (20 Newsgroups), and handwritten-digit recognition (Pendigits, 10 classes), with naive
  Bayes, boosted naive Bayes, and linear-kernel SVM as the base classifiers, and held-out train/test
  splits. Calibration is learned on labeled data and applied to held-out scores.

## Code framework

The post-hoc calibrator is a small object that sits *after* a fixed, already-trained classifier: it is
`fit` once on held-out score/label data, and thereafter `predict_proba` maps fresh raw scores to
calibrated probabilities. The substrate is only the generic machinery that already exists: arrays of
scores, labels, sorting and aggregation primitives, and a fit/apply interface for a learned calibration
map. The binary case has a single score column; the multiclass case has a matrix of class scores that
must be turned into a valid distribution. The empty slots are the map from scores to probabilities and
the rule that makes the final outputs non-negative and sum to one.

```python
import numpy as np


class CalibrationMethod:
    """Post-hoc calibrator fit on a held-out (score, label) set, applied to raw scores.

    fit(probs, labels):  probs is (n,) for binary (positive-class score) or (n, C) for
                         multiclass (rows sum to 1); labels is (n,) integer class labels.
    predict_proba(probs): returns calibrated probabilities of the same shape; the output
                          must stay a valid probability distribution (non-negative; rows
                          sum to 1 in the multiclass case).
    """

    def __init__(self):
        self.is_binary = None
        self.state_ = None

    def _fit_map(self, probs, labels):
        # TODO: the calibration map we will design.
        pass

    def _apply_map(self, fitted_map, probs):
        # TODO: apply the learned calibration map to fresh scores.
        pass

    def fit(self, probs, labels):
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels)
        if probs.ndim == 1:
            self.is_binary = True
        else:
            self.is_binary = False
        self.state_ = self._fit_map(probs, labels)
        return self

    def predict_proba(self, probs):
        probs = np.asarray(probs, dtype=float)
        calibrated = self._apply_map(self.state_, probs)
        # TODO: ensure outputs are valid probabilities.
        return calibrated
```

The `_fit_map` / `_apply_map` slots and the final validity step are the empty parts of the scaffold.
