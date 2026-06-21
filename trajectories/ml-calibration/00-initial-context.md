## Research question

A trained classifier ranks well but its reported probabilities lie. Among all inputs it rates at confidence `p`, the empirical fraction that are actually correct is not `p`; the reliability diagram bows off the diagonal. Random forests, MLPs, GBMs, and SVMs are routinely miscalibrated, each in its own direction. The design target is the **post-hoc calibration mapping**: a function that takes the frozen classifier's raw confidence and returns a well-calibrated probability, learned only from a held-out calibration set. The classifier, the data, and the train/calibrate/test splits are fixed; the contribution is the mapping itself.

## Prior art / Background / Baselines

Three existing ways to turn a score into a probability are on the table.

- **Exact empirical map.** On a fixed dataset the score is honest if `mu(s) = E[Y | f(X)=s]`, the empirical positive rate at each score. When the classifier gives a distinct score to every instance, that rate is just the instance's own 0/1 label.
- **Equal-count binning.** Sort by score, chop into a fixed number of equal-size bins, and return each bin's empirical positive rate.
- **Fitted sigmoid (Platt, 1999).** Commit to a parametric shape, `mu(s) = 1/(1+exp(A f + B))`, fit `A,B` by likelihood. Two parameters, low variance, fine on little data, and exact when the true distortion is sigmoidal.

## Fixed substrate / Code framework

Everything except the mapping is frozen and must not be touched. Four classifier-dataset pairs are run: a **Random Forest on MNIST** (10-class), an **MLP on Fashion-MNIST** (10-class), a **GBM on Madelon** (binary), and an **SVM on Breast Cancer** (binary). Each dataset is split 60% train / 20% calibrate / 20% test (stratified). The classifier is trained on the train split; the calibration mapping sees only the classifier's uncalibrated probabilities and labels on the **calibrate** split; everything is scored on the test split. The harness hands the mapping `clf.predict_proba` outputs: for binary a 1-D array of positive-class probabilities; for multiclass the full `(n, C)` softmax-like matrix whose rows sum to 1. The metrics, the binning for ECE (15 equal-width bins on the max-class confidence), the Brier and NLL computations, and the data/classifier construction are all fixed.

## Editable interface

Exactly one region is editable — the `CalibrationMethod` class in `custom_calibration.py` (lines 45–102), a `BaseEstimator` with two methods. The contract: `fit(probs, labels)` learns the mapping from the calibration split, where `probs` is `(n,)` for binary or `(n, C)` for multiclass and `labels` is `(n,)` integer class labels; `predict_proba(probs)` returns calibrated probabilities of the **same shape as the input** — for binary a 1-D array of positive-class probabilities in `[0,1]`, for multiclass a 2-D array whose rows are non-negative and sum to 1 (the harness asserts both). Available imports: `numpy`, `scipy` (`optimize`, `interpolate`, `special`), `sklearn`. Every method is a fill of this same two-method contract and nothing else.

The starting point is the scaffold default: **identity** — return the uncalibrated probabilities unchanged (clipped, and renormalized to sum to 1 for multiclass). Each candidate method replaces exactly this class body.

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

- **ECE (Expected Calibration Error)** — weighted mean of `|accuracy − confidence|` across 15 equal-width bins of the max-class confidence (for binary, of `max(p, 1−p)`).
- **Brier score** — mean squared error between the predicted probability vector and the one-hot label.
- **NLL (Negative Log-Likelihood)** — cross-entropy between predicted probabilities and true labels.

Each method is run over seeds {42, 123, 456}; the leaderboard reports per-seed values and their mean per metric per dataset. ECE measures only that confidence matches accuracy on average; Brier and NLL are the proper scores that also punish a method for being honest-on-average but wrong-per-example.
