## Research question

A trained classifier ranks well but its reported probabilities lie. Among all the inputs it rates at
confidence `p`, the empirical fraction that are actually correct is not `p` — the reliability diagram
bows off the diagonal. Random forests, MLPs, GBMs, and SVMs are all routinely miscalibrated, each in
its own direction (forests and trees pile mass near the extremes; margin classifiers do not even
output a probability). The single thing being designed is the **post-hoc calibration mapping**: a
function that takes the frozen classifier's raw confidence and returns a well-calibrated probability,
learned only from a held-out calibration set. The classifier, the data, and the train/calibrate/test
splits are all fixed; the contribution is the mapping itself.

## Prior art before the first rung

The first rung reacts to two ways of turning a score into a probability that were already on the table,
each with a flaw the calibration mapping has to dodge.

- **The exact empirical map.** On a fixed dataset there is a unique map that makes the score honest:
  `mu(s) = E[Y | f(X)=s]`, the empirical positive rate at each score. But if the classifier gives a
  distinct score to every instance, that rate at each score is just that instance's own 0/1 label, so
  the "perfect" map is a pile of 0s and 1s — maximally overconfident, useless on unseen data. *Gap:* with
  scores sparse, the empirical rate must be **aggregated** over nearby scores, and the whole question is
  *how* to aggregate without committing to the wrong shape or chasing noise.
- **Equal-count binning.** Sort by score, chop into a fixed number of equal-size bins, return each bin's
  empirical positive rate. Shape-free aggregation. *Gap:* the bin boundaries fall wherever equal counts
  put them, not where the score's meaning actually changes; a boundary in the middle of a steeply
  climbing region averages together examples that deserve different probabilities, and the bin count is a
  hyperparameter that must be cross-validated — hopeless on a small or unbalanced calibration split.
- **The fitted sigmoid (Platt, 1999).** Commit to a parametric shape: `mu(s) = 1/(1+exp(A f + B))`, fit
  `A,B` by likelihood. Two parameters, low variance, fine on little data, and exactly right when the true
  distortion is sigmoidal (SVM margins). *Gap:* it is exactly *one* shape; when the true score-to-probability
  curve is some other monotone thing — as tree/forest curves can be — the sigmoid simply cannot bend that
  way and leaves residual miscalibration.

The one belief that survives all three, robustly, dataset-independent: the true map is **monotone
non-decreasing** — the classifier ranks well, so a higher score means a not-lower probability. That is
the only prior the first rung trusts, and it is exactly what the first rung is built on.

## The fixed substrate

Everything except the mapping is frozen and must not be touched. Four classifier-dataset pairs are run:
a **Random Forest on MNIST** (10-class), an **MLP on Fashion-MNIST** (10-class), a **GBM on Madelon**
(binary), and an **SVM on Breast Cancer** (binary). Each dataset is split 60% train / 20% calibrate /
20% test (stratified). The classifier is trained on the train split; the calibration mapping sees only
the classifier's uncalibrated probabilities and labels on the **calibrate** split; everything is scored
on the test split. The harness hands the mapping `clf.predict_proba` outputs: for a binary task a 1-D
array of positive-class probabilities; for a multiclass task the full `(n, C)` softmax-like matrix whose
rows sum to 1. The metrics, the binning for ECE (15 equal-width bins on the max-class confidence), the
Brier and NLL computations, and the data/classifier construction are all in the fixed region.

## The editable interface

Exactly one region is editable — the `CalibrationMethod` class in `custom_calibration.py` (lines
45–102), a `BaseEstimator` with two methods. The contract: `fit(probs, labels)` learns the mapping from
the calibration split, where `probs` is `(n,)` for binary or `(n, C)` for multiclass and `labels` is
`(n,)` integer class labels; `predict_proba(probs)` returns calibrated probabilities of the **same shape
as the input** — for binary a 1-D array of positive-class probabilities in `[0,1]`, for multiclass a
2-D array whose rows are non-negative and sum to 1 (the harness asserts both). Available imports:
`numpy`, `scipy` (`optimize`, `interpolate`, `special`), `sklearn`. Every rung on the ladder is a fill of
this same two-method contract and nothing else.

The starting point is the scaffold default: **identity** — return the uncalibrated probabilities
unchanged (clipped, and renormalized to sum to 1 for multiclass). Each later method replaces exactly
this class body.

```python
# EDITABLE region of custom_calibration.py (lines 45-102) - default fill (no calibration)
class CalibrationMethod(BaseEstimator):
    """Post-hoc probability calibration method.

    For binary classification, fit() receives positive-class probabilities (n,);
    for multiclass, the full probability matrix (n, C). predict_proba returns
    calibrated probabilities of the same shape (binary: 1-D in [0,1];
    multiclass: 2-D, rows sum to 1)."""

    def __init__(self):
        self.is_binary = None

    def fit(self, probs, labels):
        """Default: identity (no calibration)."""
        if probs.ndim == 1:
            self.is_binary = True
        else:
            self.is_binary = False
        return self

    def predict_proba(self, probs):
        """Default: return uncalibrated probabilities unchanged."""
        if self.is_binary:
            return np.clip(probs, 0, 1)
        else:
            probs = np.clip(probs, 1e-15, 1.0)
            probs = probs / probs.sum(axis=1, keepdims=True)   # rows must sum to 1
            return probs
```

## Evaluation settings

Three metrics, **all lower is better**, computed on the test split after calibration:

- **ECE (Expected Calibration Error)** — weighted mean of `|accuracy − confidence|` across 15
  equal-width bins of the max-class confidence (for binary, of `max(p, 1−p)`).
- **Brier score** — mean squared error between the predicted probability vector and the one-hot label
  (a strictly proper scoring rule; range `[0,1]` binary, `[0,2]` multiclass).
- **NLL (Negative Log-Likelihood)** — cross-entropy between predicted probabilities and true labels
  (the other strictly proper scoring rule).

Each method is run over seeds {42, 123, 456}; the leaderboard reports per-seed values and their mean
per metric per dataset. ECE measures only that confidence matches accuracy on average; Brier and NLL are
the proper scores that also punish a method for being honest-on-average but wrong-per-example.
