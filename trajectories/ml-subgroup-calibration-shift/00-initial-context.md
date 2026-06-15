## Research question

A binary classifier is already trained and frozen; it hands me a positive-class probability `p` for
every example, and those numbers are not honest — among the examples it labels `0.7`, the true positive
rate is not `0.7`. I want them honest **inside every subgroup**, not just on average: a calibrator that
looks perfect pooled across the whole population can still be badly over-confident on one protected
subgroup and under-confident on another, and the metric that bites is the *worst* subgroup's calibration
error, not the mean. The one thing I get to design is the **calibration mapping** applied post-hoc to
those positive-class probabilities, optionally using a subgroup id. Everything upstream — the data, the
shifted splits, the base classifier — is fixed.

The twist that makes this hard is the shift. For each dataset a domain score peels off a held-out tail
for the test set; calibration is fit on the source region and evaluated on the shifted region. So a map
learned on the calibration split is judged on a distribution it never saw, and any per-group machinery
that overfits a small calibration sample will be punished when that sample's quirks fail to transfer.

## Prior art before the first rung (the calibration-map lineage)

The methods on this ladder all answer the same sub-question — given a held-out labeled calibration set,
what *family* of maps `p ↦ q` do I fit? — and they react to a line of older choices that the rungs below
either reuse or reject.

- **Platt scaling (Platt 1999).** Squash the score through a fitted sigmoid `q = σ(a·z + b)` on the
  logit `z = logit(p)`, two parameters by maximum likelihood. Cheap, low-variance, the right cost. But
  it is exactly one shape: it can only *spread* scores toward the extremes, never *gather* them, and a
  nonzero intercept `b` moves the decision boundary so it can change the predicted class. Gap: one rigid
  shape, and it can disturb accuracy.
- **Histogram / equal-count binning (Zadrozny & Elkan 2001).** Chop the scores into bins and report each
  bin's empirical positive rate. Shape-free, fixing the sigmoid's rigidity. But the bin boundaries fall
  wherever equal counts put them, not where the score changes meaning, and the bin count needs
  cross-validation — hopeless on a small, shifted calibration split. Gap: arbitrary boundaries, a knob
  to tune.
- **Maximum-likelihood as the fitting objective.** Every parametric rung here fits by minimizing
  negative log-likelihood (binary cross-entropy), because NLL is a proper scoring rule — in expectation
  it is minimized exactly when the reported probability equals the true conditional probability — while
  the reported metric, binned ECE, is non-differentiable and only measured, never optimized.
- **Empirical-Bayes / James–Stein shrinkage (Stein 1956; James & Stein 1961; Efron & Morris 1973).**
  When many related parameters are each estimated from their own small sample, the per-coordinate MLE is
  inadmissible; pulling every estimate toward a common center, harder for the noisier ones, strictly
  reduces total risk. This is the regularizer the subgroup-aware rung reaches for.

## The fixed substrate

The pipeline is frozen and must not be touched. Three cached high-stakes tabular datasets from AIF360 —
**Adult** (census income; subgroups sex×race), **COMPAS** (recidivism risk; subgroups race×sex),
**Law School GPA** (binarized at median first-year GPA; subgroups race×gender). For each, a domain score
sorts each class and sends the tail to the test split, fitting calibration on the source region and
evaluating on the shifted tail. The base classifier is a `StandardScaler → LogisticRegression`
(`class_weight="balanced"`, `max_iter=1200`) pipeline; its `predict_proba(...)[:, 1]` is the
positive-class probability stream that enters the calibrator. Subgroups are the cross-product of the
protected attributes, integer-coded. The harness exposes `numpy`, `scipy.optimize`, `scipy.special`,
`sklearn.isotonic.IsotonicRegression`, and `sklearn.linear_model.LogisticRegression`. Note that the loop
always passes a **single positive-class probability vector** (1-D `probs`) — this is a binary problem
end to end; there is no multiclass path to fill.

## The editable interface

Exactly one region is editable — the `CalibrationMethod` class in
`scikit-learn/custom_subgroup_calibration.py` (lines 200–219). Every method on the ladder is a fill of
this same contract: `fit(probs, labels, groups=None)` learns the map from the calibration split, and
`predict_proba(probs, groups=None)` returns calibrated positive-class probabilities in `[0, 1]`. `probs`
is `(n,)` raw positive-class probabilities, `labels` is `(n,)` in `{0, 1}`, `groups` is `(n,)` integer
subgroup ids and may be ignored by group-agnostic methods. The output must be valid probabilities.

The starting point is the scaffold default: the **identity** map (clip into `[ε, 1−ε]` and return).
Each rung replaces exactly this class and nothing else.

```python
# EDITABLE region of custom_subgroup_calibration.py (lines 200-219) — default fill (identity)
class CalibrationMethod:
    """Editable calibration method.

    Implement fit() and predict_proba() to map raw positive-class probabilities
    to calibrated positive-class probabilities.
    """

    def __init__(self):
        self.eps = 1e-6
        self._identity = True

    def fit(self, probs, labels, groups=None):
        probs = np.asarray(probs).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)
        self._base_rate = float(np.clip(labels.mean(), self.eps, 1.0 - self.eps))
        return self

    def predict_proba(self, probs, groups=None):
        probs = np.asarray(probs).reshape(-1)
        return np.clip(probs, self.eps, 1.0 - self.eps)
```

## Evaluation settings

Three datasets — **Adult**, **COMPAS**, **Law School GPA** — each a single seed (42). Four metrics per
dataset: **`worst_group_ece`** (worst-subgroup expected calibration error, 15 equal-width confidence
bins; **lower is better**, the primary objective), **`brier`** (Brier score on the test split; lower is
better), **`max_subgroup_gap`** (max over subgroups of `|accuracy − mean confidence|`, here the spread
between the worst and best subgroup ECE; lower is better), and **`subgroup_auroc`** (mean subgroup-level
AUROC; higher is better, reported diagnostically and unaffected by any monotone calibration map).
