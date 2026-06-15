# Context: selective prediction and group disparities for fixed tabular classifiers

## Research question

A base classifier and its train/calibration/test pipeline are fixed and not editable. The only
thing left to design is the *acceptance rule*: given a target coverage (say, predict on 80% of
test points and defer the rest to a downstream human reviewer or backup process), decide which
test examples are accepted and which are deferred. Errors on accepted points are costly;
deferrals are manageable but not free, since each one consumes a reviewer.

The data carry a subgroup variable `g ∈ {1, …, k}` — for high-stakes tabular problems these are
demographic or clinically meaningful subpopulations (sex, race, presence of a support device).
The base classifier never consumes `g`; the post-hoc acceptance policy may receive subgroup ids
from the calibration and evaluation harness. Under subgroup shift the difficulty is that
confidence is not uniformly reliable across subgroups, so an acceptance rule tuned only for
average behavior can quietly mistreat one subgroup.

A good rule must, at the target coverage, simultaneously: (1) keep *selective risk* (error on
the accepted points) low; (2) keep *worst-subgroup* selective risk low, not just the average;
(3) avoid concentrating deferrals on one subgroup, i.e. keep the gap between the most-deferred
and least-deferred subgroup small; (4) preserve the AUROC of its acceptance score as a
correctness-ranking signal; and (5) be cheap enough to fit and apply offline on calibration data
with modest compute, without retraining the base model. The naive single global confidence
threshold satisfies (1) on average but can fail (2) and (3) badly. Closing that gap is the
problem.

## Background

**Selective prediction / classification with a reject option.** Letting a fixed classifier
abstain when it is unsure is an old idea. Chow (1957, 1970) derived the optimal reject rule: for
a known posterior and a fixed misclassification-versus-rejection cost, the Bayes-optimal policy
abstains exactly when the maximum class posterior `max_y P(y|x)` falls below a threshold, and
predicts otherwise. So thresholding a confidence score *is* the classical optimal abstention
mechanism; the only questions are which score to use and where to put the threshold.

**The risk-coverage / accuracy-coverage curve.** El-Yaniv & Wiener (2010) formalized evaluating
such a classifier by its *coverage* (the fraction of points it predicts on) and its *selective
risk* (the error rate on the predicted subset). Sweeping the threshold traces out a
risk-coverage curve; lowering coverage should, for a good confidence score, lower selective risk.
This curve is the standard yardstick for any abstention rule.

**Softmax response (SR).** For modern classifiers that output a softmax `P̂(y|x)`, the dominant
confidence score is the maximum softmax probability `P̂(ŷ|x)` of the predicted class
`ŷ(x) = argmax_y P̂(y|x)` (Hendrycks & Gimpel, 2017; Geifman & El-Yaniv, 2017). One often
reports it in a logit-transformed form, the *confidence*

```
ĉ(x) = ½ · log( P̂(ŷ|x) / (1 − P̂(ŷ|x)) )            (binary)
ĉ(x) = ½ · log( P̂(ŷ|x) / (1 − P̂(ŷ|x)) ) + ½ · log(k − 1)   (k classes)
```

which is non-negative because `P̂(ŷ|x) ≥ 1/k`, equals 0 at the maximally-unsure prediction, and
reduces to the binary form at `k = 2`. Crucially `ĉ(x)` is a *monotone* transform of the
maximum softmax probability, so the relative ranking of points — and therefore the entire
accuracy-coverage curve — is unchanged whether one thresholds the logit or the raw probability.
Across a broad range of applications, more confident SR predictions are empirically more
accurate (Hanczar et al. 2008; Geifman & El-Yaniv 2017), which is what makes SR a usable score.

**The margin distribution.** It is convenient to summarize a confidence-based selective
classifier `(ŷ, ĉ)` by its *margin*: `m(x) = ĉ(x) ≥ 0` on correct predictions (`ŷ(x) = y`) and
`m(x) = −ĉ(x) ≤ 0` on incorrect ones. At a confidence threshold `τ`, a predicted point is
correct exactly when `m ≥ τ`, the classifier abstains when `−τ < m < τ`, and a point is
predicted-but-wrong when `m ≤ −τ`. If `F` is the margin CDF, the selective accuracy and coverage
at `τ` are

```
A_F(τ) = (1 − F(τ)) / ( F(−τ) + 1 − F(τ) ),     coverage(τ) = 1 − F(τ) + F(−τ).
```

The whole accuracy-coverage curve is a functional of `F` alone, so questions about *which* group
selective classification helps reduce to comparing each group's margin distribution.

**Diagnostic empirical finding: confidence is not uniformly informative across subgroups.** On
high-stakes tabular and vision/NLP data where models latch onto a spurious correlation, the
subgroups for which the spurious cue fails are systematically harder. Their margin distributions
are shifted left relative to the average: they hold disproportionately many *confident but
incorrect* examples, and on the worst subgroups confidence can even be *anticorrelated* with
correctness. So a confidence score that ranks correctness well on average can rank it poorly, or
backwards, on a particular subgroup — a pre-method fact about existing classifiers, observed
before any new acceptance rule.

**Equalized odds.** Hardt, Price & Srebro (2016) formalized a fairness criterion for a decision
rule: equal true-positive and false-positive rates across groups. Viewing "predict vs. abstain"
as a meta-classification problem — a true positive is *predict-and-correct*, a false positive is
*predict-and-wrong* — gives a natural notion of equalized odds for an abstention rule: the
probability of accepting should match across groups separately among points the base classifier
gets right and among points it gets wrong.

**Training-time alternative.** A different route closes subgroup gaps by *training*: group
distributionally-robust optimization (Sagawa, Koh, Hashimoto & Liang, 2020) minimizes the
worst-group loss `max_g Ê[ℓ(θ;(x,y))|g]` using group labels at training time, yielding more
similar full-coverage accuracies across groups. This needs the base model to be retrained and
group labels at training time — outside the scope of an acceptance-rule-only contribution — but
it frames the same goal from the model side rather than the decision side.

## Baselines

These are the acceptance rules a new rule would be measured against.

**Global confidence thresholding (Chow / softmax response).** Take the score `s(x) = max_y
P̂(y|x)` (equivalently the logit confidence `ĉ`) and pick one global threshold so that the
overall acceptance rate equals the target coverage — i.e. the threshold is the
`(1 − target_coverage)` quantile of `s` over calibration. Accept iff `s(x) ≥ threshold`. This is
Chow's optimal rule for a fixed cost and is the canonical SR selective classifier. **Limitation:**
one threshold is applied to every point regardless of subgroup. Because the worst subgroup's
margin distribution is shifted left, at a fixed confidence threshold a smaller fraction of its
points can clear the bar — so it is covered less and deferred more than other subgroups, and the
deferral burden lands unevenly. Worse, on a subgroup whose full-coverage accuracy is below half,
raising the threshold can move its selective accuracy in the wrong direction. The rule optimizes
an average and is blind to which subgroup pays.

**Split-conformal abstention.** Compute non-conformity scores on calibration data and choose the
threshold so the achieved coverage is at least the target with high probability (Vovk, Gammerman
& Shafer, 2005). This adds a finite-sample coverage guarantee over the global thresholding rule.
**Limitation:** the guarantee is *marginal* — a single threshold calibrated over the pooled
calibration set. It controls aggregate coverage but says nothing about how coverage is split
across subgroups, so under subgroup shift it can still cover one subgroup far less than another.

**Learned deferral / learning-to-defer.** Train a compact meta-classifier to predict whether the
base model will be correct on a given example, and defer the ones it predicts as wrong (related:
Mozannar & Sontag, 2020; Madras et al., 2018). **Limitation:** it learns a single
correctness-predictor over all data and, like the score it sits on, can inherit the same
subgroup-dependent miscalibration; it adds a model to fit and gives no direct handle on each
subgroup's deferral rate.

**The line where these stall.** Every rule above chooses one decision boundary that is shared
across subgroups. Where the confidence signal behaves differently from one subgroup to the next —
which is exactly the subgroup-shift regime — a single shared boundary cannot hold each subgroup
to the same coverage; some subgroup is systematically over-deferred. Getting past that point is
where the prior art stalls.

## Evaluation settings

The natural yardsticks are cached high-stakes tabular datasets from AIF360, each with one or more
subgroup attributes; a group is formed for each combination of label and a spuriously-correlated
attribute:

- **Adult** — Census income prediction; subgroup attributes: sex, race.
- **COMPAS** — ProPublica recidivism risk; subgroup attributes: race, sex.
- **Law School GPA** — admissions/outcome data, binarized at the training-set median; subgroup
  attributes: race, gender.

Each dataset is split into train / calibration / test. The base classifier is trained on train;
the acceptance rule is *fit on calibration* probabilities, labels, and subgroup ids; and it is
*evaluated on test*. The base classifier and the splits are fixed.

Metrics at a target coverage (here 80%): selective risk on accepted points (lower better);
worst-subgroup selective risk among accepted points (lower better); deferral-rate gap, the
maximum-minus-minimum subgroup deferral rate (lower better); and AUROC of the acceptance score
as a predictor of correctness (higher better). The accuracy-coverage (risk-coverage) curve,
traced by sweeping the threshold, is the underlying object behind these numbers.

## Code framework

The acceptance rule plugs into a fixed offline pipeline: the base classifier has already produced
calibration-time class probabilities, and a held-out test set will get the same probabilities.
The rule is a small object that is *fit* on calibration `(probs, y_true, groups)` and then
*applies* an accept/defer decision on test points. Nothing about how that decision is made is
settled — that rule is exactly what is to be designed — so the substrate is only the generic
selective-policy harness that already exists: an object that exposes `fit`, an `acceptance_score`,
a boolean `predict_accept`, and a `calibration_summary`, parameterized by a target coverage.

```python
import numpy as np

TARGET_COVERAGE_DEFAULT = 0.8


class SelectivePolicy:
    """Acceptance rule for a fixed base classifier. Fit on calibration probabilities,
    labels, and subgroup ids; then accept/defer test points at a target coverage.
    The base classifier and the train/calibration/test split are fixed."""

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT,
                 random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        # TODO: calibration state

    def fit(self, probs: np.ndarray, y_true: np.ndarray,
            groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        # probs: (n, n_classes) calibration-time base-model probabilities
        # y_true: (n,) calibration labels
        # groups: (n,) integer subgroup ids
        # TODO: fit policy state from calibration data
        return self

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray,
                         X: np.ndarray | None = None) -> np.ndarray:
        # Higher score = more confident -> more likely to accept.
        # TODO: the confidence score the rule ranks points by.
        raise NotImplementedError

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray,
                       X: np.ndarray | None = None) -> np.ndarray:
        # Boolean array: True = accept, False = defer.
        # TODO: turn scores + calibrated state into accept/defer decisions.
        raise NotImplementedError

    def calibration_summary(self) -> dict[str, float]:
        # TODO: report whatever the calibration produced.
        return {}
```

The single empty slot is the acceptance rule itself: which confidence score to rank by, and how
to turn that score plus calibration data into accept/defer decisions that hit the target coverage.
