# Isotonic regression calibration, distilled

Isotonic regression calibration is a non-parametric post-hoc method that maps a classifier's raw
scores to calibrated probabilities by fitting the best **monotonically non-decreasing** function from
score to label, using the **pool-adjacent-violators algorithm (PAVA)**. It sits between the rigid
two-parameter sigmoid (Platt scaling) and shape-free equal-count binning: it assumes only that higher
score means higher probability, so it can fix any monotone distortion, and its block boundaries are
chosen by the data rather than fixed in advance. Multiclass is handled one-against-all per class,
followed by renormalization into a valid distribution.

## Problem it solves

A fixed, already-trained classifier produces scores `s(x)` that **rank** examples well but are **not
calibrated**: among examples scored near `p`, the true positive fraction is not near `p`. Downstream
decision-making (cost-sensitive choices, combining classifiers, expected-value computation) needs the
scores to be actual probabilities. From a held-out labeled calibration set, learn a map
`s(x) ↦ p̂(x)` that returns calibrated probabilities (non-negative; summing to one across classes for
multiclass) without retraining the base model and without disturbing the ranking.

## Key idea

The only robust prior is **monotonicity** — the classifier ranks correctly, so the score-to-probability
map is non-decreasing. Restrict the calibration map to the class of all non-decreasing functions and fit
it by order-constrained least squares. Sort the calibration pairs by score, `s_1 ≤ … ≤ s_n` with binary
targets `g_i ∈ {0,1}`, and solve

```
minimize  Σ_i w_i (g_i − ĝ_i)²   subject to   ĝ_1 ≤ ĝ_2 ≤ … ≤ ĝ_n.
```

Why this gives a probability: the single constant minimizing `Σ w_i (g_i − c)²` over a set of points is
the weighted mean `(Σ w_i g_i)/(Σ w_i)`, which for 0/1 targets is the empirical positive rate. The
optimal monotone fit is **piecewise constant**, and on each block its value is the block's weighted
average — an honest empirical probability in `[0,1]`. Squared error and Bernoulli log-loss are both
Bregman losses for the mean; for this order-restricted mean problem, replacing squared error by any such
loss leaves the fitted vector unchanged. The squared-error solution is therefore the same constrained
calibration obtained under log-loss.

## The solution and PAVA

The optimum has the closed-form **min-max characterization**

```
ĝ_i = min_{ℓ ≥ i}  max_{k ≤ ℓ}  ( Σ_{j=k}^{ℓ} w_j g_j ) / ( Σ_{j=k}^{ℓ} w_j ),
```

equivalently: a partition into maximal blocks with `ĝ` constant on each block, equal to that block's
weighted mean, with strictly increasing block values. (It is the left-hand slopes of the greatest
convex minorant of the cumulative-sum diagram `(Σ_{j≤k} w_j, Σ_{j≤k} w_j g_j)`.)

Computing the min-max directly is slow. **Pool Adjacent Violators (PAVA)** computes the same unique
optimum in one pass:

```
sort by score; each point is its own block with value g_i, weight w_i
scan left to right:
    if an adjacent pair violates monotonicity (value_L > value_R):
        pool: merge into one block, value = (w_L·v_L + w_R·v_R)/(w_L + w_R), weight = w_L + w_R
        back-merge with the left neighbor while that creates a new violation
return the block-constant fit
```

Correctness: maintain the exact isotonic least-squares solution for each processed prefix. Appending a
new singleton can only violate the last active boundary; when the last two active means are reversed,
projecting their two-block unconstrained optimum onto `z_L ≤ z_R` sets both values to their weighted
mean, exactly the pooling step. If that pooled value violates the previous boundary, repeat the same
projection to the left. When no adjacent active boundary is violated, the prefix optimum is restored; by
induction, the final stack is the unique global optimum. Complexity `O(n)` after the initial sort
(`O(n log n)` including the sort). The result is a **data-chosen binning** — coarse where the classifier
ranks poorly (many violations pooled away), fine where it ranks well — which is exactly the arbitrariness
of fixed equal-count binning repaired, with no bin count to cross-validate.

## Prediction

Before fitting, tied scores are aggregated by averaging their targets and summing their weights. PAVA
yields fitted values at the sorted score thresholds. The standard sklearn predictor trims redundant
thresholds and **interpolates linearly** between adjacent threshold values. A test score outside the
calibration range is **clipped in input space** to the nearest fitted score endpoint, and `y_min=0`,
`y_max=1` bound the fitted values to `[0,1]`.

## Multiclass

Do **not** calibrate the joint `(k−1)`-dimensional simplex map (no order to impose; curse of
dimensionality). Reduce **one-against-all**: for each class `c`, calibrate the class-`c` score against
the binary target `1{label = c}` with its own isotonic regression, giving `r_c(x) ≈ P(c | x)`. Because
each one-against-all output already estimates `P(c | x)`, reconcile the `k` independent estimates into a
distribution by **normalization** (the simplest valid combiner for one-against-all, preferable to
least-squares or log-loss coupling here): floor each at a tiny value to avoid a degenerate all-zero row,
then divide by the row sum:

```
p̂(c | x) = clip(r_c(x), ε, ∞) / Σ_{c'} clip(r_{c'}(x), ε, ∞),   ε = 1e-15.
```

## When to use it

Isotonic calibration is non-parametric, with as many effective degrees of freedom as there are blocks,
so it needs more calibration data than the two-parameter sigmoid to avoid overfitting. Below that, or on
highly imbalanced data, the sigmoid's rigidity can be safer. Quality is judged with strictly proper
scoring rules — Brier score (MSE to the one-hot label) and negative log-likelihood — alongside
reliability diagrams.

## Working code

A compact implementation using `sklearn.isotonic.IsotonicRegression`:

```python
import numpy as np
from sklearn.base import BaseEstimator


class CalibrationMethod(BaseEstimator):
    """Isotonic Regression calibration: a non-parametric, monotonically non-decreasing
    map from uncalibrated scores to calibrated probabilities, fit by pool-adjacent-violators.
    Multiclass is one-against-all per class, then renormalized."""

    def __init__(self, eps=1e-15):
        self.is_binary = None
        self.state_ = None
        self.eps = eps

    def _new_binary_map(self):
        # IsotonicRegression == PAVA: minimize sum_i (g_i - ghat_i)^2 with ghat non-decreasing
        # in the score; each block value is the weighted mean of the 0/1 targets = empirical
        # P(positive | score). It sorts X, averages duplicate X values, trims redundant
        # thresholds, predicts by linear interpolation, and with out_of_bounds="clip"
        # clamps out-of-range test scores to the fitted score domain.
        from sklearn.isotonic import IsotonicRegression as IR
        return IR(out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True)

    def _fit_map(self, probs, labels):
        if probs.ndim == 1:                      # binary: a single positive-class score column
            iso = self._new_binary_map()
            iso.fit(probs, labels.astype(float))
            return [iso]

        maps = []                                # multiclass: one-against-all per class
        for c in range(probs.shape[1]):
            binary_labels = (labels == c).astype(float)
            iso = self._new_binary_map()
            iso.fit(probs[:, c], binary_labels)
            maps.append(iso)
        return maps

    def _apply_map(self, fitted_map, probs):
        if self.is_binary:
            calibrated = fitted_map[0].predict(probs)          # linear interpolation + clip
            return np.clip(calibrated, 0.0, 1.0)

        calibrated = np.zeros_like(probs, dtype=float)
        for c, iso in enumerate(fitted_map):
            calibrated[:, c] = iso.predict(probs[:, c])        # per-class r_c(x)
        calibrated = np.clip(calibrated, self.eps, None)       # floor before normalization
        return calibrated / calibrated.sum(axis=1, keepdims=True)

    def fit(self, probs, labels):
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels)
        self.is_binary = probs.ndim == 1
        self.state_ = self._fit_map(probs, labels)
        return self

    def predict_proba(self, probs):
        probs = np.asarray(probs, dtype=float)
        return self._apply_map(self.state_, probs)
```

For reference, the core active-block PAVA loop (single-pass `O(n)` on sorted data, back-merging on each
pooled block):

```python
import numpy as np

def pav(y, w=None):
    """Pool-adjacent-violators: argmin_g sum_i w_i (y_i - g_i)^2 s.t. g non-decreasing.
    Assumes y is already sorted by the covariate (here, by the classifier score)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    starts, ends, wt, val = [], [], [], []
    for i, (yi, wi) in enumerate(zip(y, w)):
        starts.append(i)
        ends.append(i + 1)
        wt.append(wi)
        val.append(yi)
        while len(val) >= 2 and val[-2] > val[-1]:                # adjacent violation -> pool
            new_w = wt[-2] + wt[-1]
            new_v = (wt[-2] * val[-2] + wt[-1] * val[-1]) / new_w
            new_start, new_end = starts[-2], ends[-1]
            starts[-2:] = [new_start]
            ends[-2:] = [new_end]
            wt[-2:] = [new_w]
            val[-2:] = [new_v]
    out = np.empty(n, dtype=float)
    for start, end, value in zip(starts, ends, val):
        out[start:end] = value
    return out
```
