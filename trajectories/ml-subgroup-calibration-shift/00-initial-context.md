## Research question

A frozen binary classifier hands me a positive-class probability `p` for every example, and those probabilities are not honest — among the examples it labels `0.7`, the true positive rate is not `0.7`. I want them honest inside every subgroup, not just on average: a calibrator that looks perfect pooled across the whole population can still be over-confident on one protected subgroup and under-confident on another. The metric that matters is the worst subgroup's calibration error. The only thing I get to design is the post-hoc calibration mapping applied to those probabilities, optionally using a subgroup id. Everything upstream — the data, the shifted splits, and the base classifier — is fixed.

The difficulty is the shift. A domain score peels off a held-out tail for the test set; calibration is fit on the source region and evaluated on the shifted tail. A map learned on the calibration split is judged on a distribution it never saw, and any per-group machinery that overfits a small calibration sample will be punished when those quirks fail to transfer.

## Prior art / Background / Baselines

The methods on this ladder answer the same sub-question — given a held-out labeled calibration set, what family of maps `p ↦ q` do I fit? — and react to older choices that the rungs below either reuse or reject.

- **Platt scaling (Platt 1999).** It squashes the score through a fitted sigmoid `q = σ(a·z + b)` on the logit `z = logit(p)`, two parameters fit by maximum likelihood. Gap: one rigid shape — it can only spread scores toward the extremes, never gather them — and a nonzero intercept moves the decision boundary, so it can change the predicted class.
- **Histogram / equal-count binning (Zadrozny & Elkan 2001).** It chops the scores into bins and reports each bin's empirical positive rate. Gap: the bin boundaries fall wherever equal counts put them, not where the score changes meaning, and the bin count needs tuning that is unreliable on a small, shifted calibration split.
- **Maximum-likelihood fitting.** Parametric methods here minimize negative log-likelihood, because NLL is a proper scoring rule — in expectation it is minimized exactly when the reported probability equals the true conditional probability — while binned ECE is non-differentiable and only measured, never optimized.
- **Empirical-Bayes / James–Stein shrinkage (Stein 1956; James & Stein 1961; Efron & Morris 1973).** When many related parameters are each estimated from their own small sample, the per-coordinate MLE is inadmissible; pulling every estimate toward a common center, harder for the noisier ones, strictly reduces total risk.

## Fixed substrate / Code framework

The pipeline is frozen. Three cached tabular datasets from AIF360 are used: **Adult** (census income; subgroups sex×race), **COMPAS** (recidivism risk; subgroups race×sex), and **Law School GPA** (binarized at median first-year GPA; subgroups race×gender). For each, a domain score sorts each class and sends the tail to the test split, fitting calibration on the source region and evaluating on the shifted tail. The base classifier is a `StandardScaler → LogisticRegression` pipeline (`class_weight="balanced"`, `max_iter=1200`); its `predict_proba(...)[:, 1]` is the positive-class probability stream that enters the calibrator. Subgroups are the cross-product of the protected attributes, integer-coded. The harness exposes `numpy`, `scipy.optimize`, `scipy.special`, `sklearn.isotonic.IsotonicRegression`, and `sklearn.linear_model.LogisticRegression`. The loop always passes a single positive-class probability vector (1-D `probs`); this is a binary problem end to end.

## Editable interface

Only one region is editable: the `CalibrationMethod` class in `scikit-learn/custom_subgroup_calibration.py` (lines 200–219). Every method on the ladder fills the same contract: `fit(probs, labels, groups=None)` learns the map from the calibration split, and `predict_proba(probs, groups=None)` returns calibrated positive-class probabilities in `[0, 1]`. `probs` is `(n,)` raw positive-class probabilities, `labels` is `(n,)` in `{0, 1}`, and `groups` is `(n,)` integer subgroup ids and may be ignored by group-agnostic methods. The output must be valid probabilities.

The starting point is the scaffold default: the identity map (clip into `[ε, 1−ε]` and return). Each rung replaces exactly this class and nothing else.

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

Three datasets — **Adult**, **COMPAS**, **Law School GPA** — each with a single seed (42). Four metrics per dataset: **`worst_group_ece`** (worst-subgroup expected calibration error, 15 equal-width confidence bins; lower is better, the primary objective), **`brier`** (Brier score on the test split; lower is better), **`max_subgroup_gap`** (max over subgroups of `|accuracy − mean confidence|`, here the spread between the worst and best subgroup ECE; lower is better), and **`subgroup_auroc`** (mean subgroup-level AUROC; higher is better, reported diagnostically and unaffected by any monotone calibration map).
