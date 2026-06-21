A classifier we have already trained — naive Bayes, a boosted model, a linear-kernel SVM, it does not matter which — hands us a real-valued score $s(x)$ for each example, and the trouble is precise: those scores *rank* examples well but are *not calibrated*. When I bin the calibration scores and plot, for each bin, the score against the empirical fraction of positives in that bin, the points do not lie on the diagonal $y=x$. Naive Bayes bows away from it because the conditional-independence assumption multiplies correlated evidence as though it were independent and over-counts, so a reported posterior of $0.97$ might correspond to a true rate of $0.8$. An SVM is worse in kind: its "score" is a signed distance to the hyperplane, not even an attempt at a probability, and even after squashing it into $[0,1]$ by $s = f/(2a) + 1/2$ the empirical-rate-versus-score curve is some S-shape rather than a line. Yet in both cases the points still march upward — higher score, higher empirical positive rate, monotonically. The ranking is right; only the vertical axis is wrong. So the task is not to reorder anything but to learn, from a held-out labeled calibration set, a map from the score the classifier gives me to the probability it should have given me. I have pairs $(s(x_i), g_i)$ with $g_i \in \{0,1\}$, and I want $m(s) \approx P(c \mid s(x)=s)$, the expected label at that score. Reading that off directly — averaging the labels at each distinct score — fails because in real data the distinct scores are far more numerous than the labeled examples, so almost every score carries one or two examples and its "empirical rate" is just that example's $0$ or $1$. I have to aggregate nearby scores, and the entire question is *how*.

Two ways to aggregate are already on the table and each has a flaw I can name. Platt scaling commits to a shape — it fits a two-parameter sigmoid $\hat p = 1/(1+\exp(A\,s+B))$ by maximizing the calibration-label likelihood — and when the true distortion really is sigmoidal, as SVM margins often are, it nails it with beautiful regularization; but it is exactly two parameters of exactly one shape, so when the true curve is some other monotone thing, as naive-Bayes curves can be, the sigmoid cannot bend that way and whatever it cannot represent stays as residual miscalibration. Equal-count binning swings to the other extreme — sort by score, chop into $b$ bins of equal size, return each bin's positive fraction — which is shape-free, but its boundaries fall wherever equal counts put them rather than where the score changes meaning, so a boundary landing in a region where the true probability climbs steeply averages together examples that plainly deserve different probabilities; and $b$ must be cross-validated, which is hopeless on a small or unbalanced calibration set. Parametric is too rigid, free-form binning too arbitrary. I want something between them.

I propose isotonic regression calibration. The one thing I believe robustly about the true map, independent of any dataset, is that it is *non-decreasing* — that is exactly what the reliability plots assert when they bow off the diagonal while still climbing. So instead of committing to a sigmoid or to nothing, I restrict the calibration map to the class of *all* non-decreasing functions and fit the best one to the calibration labels by order-constrained least squares. Sorting the pairs by score, $s_1 \le s_2 \le \dots \le s_n$ with targets $g_1, \dots, g_n$ and weights $w_i$ (unit for now, but kept because tied scores will be merged with weight equal to their count), I solve
$$\min_{\hat g}\ \sum_i w_i\,(g_i - \hat g_i)^2 \quad\text{subject to}\quad \hat g_1 \le \hat g_2 \le \dots \le \hat g_n.$$
Squared error is not an arbitrary choice of metric here; it is the choice that makes the answer an honest probability. Drop the constraint and ask what single constant $c$ minimizes $\sum_i w_i (g_i - c)^2$ over a set of points: differentiating gives $-2\sum_i w_i(g_i - c) = 0$, so $c = (\sum_i w_i g_i)/(\sum_i w_i)$, the weighted mean, which for $0/1$ targets is exactly the empirical positive rate. Wherever the order constraint forces the fit to be constant on a group — and it does, on every group whose raw labels disagree with monotonicity — the value it takes there is that group's empirical positive rate, living in $[0,1]$, precisely the calibrated probability I wanted. And I am not wedded to squared error: if I had written cross-entropy $\sum_i w_i[-g_i\log c - (1-g_i)\log(1-c)]$ and differentiated, $-\sum_i w_i g_i/c + \sum_i w_i (1-g_i)/(1-c) = 0$ solves to the same weighted mean. Squared error and Bernoulli log-loss are both Bregman losses for the mean, and in this order-restricted mean problem any such loss chooses the same block means, so the same constrained calibration is obtained either way; I solve the squared-error form because it is the cleanest quadratic.

The structure of the optimum hands me the algorithm. The solution is piecewise constant: on any maximal run of constant fitted value — a block — the order constraints at its interior are active equalities while the constraints at its ends are slack, so the block is free to sit at its unconstrained minimizer, the weighted mean of the targets it covers. The whole problem reduces to finding the right partition into blocks; the values are then forced. There is a closed-form min-max characterization,
$$\hat g_i = \min_{\ell \ge i}\ \max_{k \le \ell}\ \frac{\sum_{j=k}^{\ell} w_j g_j}{\sum_{j=k}^{\ell} w_j},$$
equivalently the left-hand slopes of the greatest convex minorant of the cumulative-sum diagram $(\sum_{j\le k} w_j,\ \sum_{j\le k} w_j g_j)$ — but computing it literally is an $O(n^2)$ double loop. The local move is far cheaper. Walk left to right, each point its own block. Whenever an adjacent pair violates monotonicity, with left value $v_L$ exceeding right value $v_R$, the two-block unconstrained optimum $(v_L, v_R)$ lies outside the feasible halfspace $z_L \le z_R$; the weighted least-squares projection onto that halfspace lands on the boundary $z_L = z_R$ at the pooled value $(w_L v_L + w_R v_R)/(w_L + w_R)$. So I pool the pair into a single block with that value and combined weight. The subtlety is that pooling can lower the new block below its left neighbor, creating a fresh upstream violation, so after each pool I turn around and back-merge to the left until the active boundaries are again ordered before resuming. This pool-adjacent-violators procedure computes the same unique global optimum, which I can prove by prefixes: maintain the exact isotonic least-squares fit for the processed prefix as a stack of blocks holding total weight and weighted sum; appending a singleton either respects the last boundary (feasible, so still optimal) or reverses the last two means, in which case the two-block projection is exactly the pool, and if that pooled value reverses the boundary to its left the same projection applies again; when no active boundary is violated the prefix optimum is restored, and induction to $n$ gives the global optimum. Each original point is pushed once and merges pop from the end, so it is $O(n)$ after the $O(n\log n)$ sort.

The payoff over binning is that the resulting intervals — a set of blocks, each with a constant probability — *are* a binning, but one whose boundaries and sizes are chosen by the data: many examples get pooled together in regions where the classifier ranks badly (many violations to fix), few or none where it ranks well (no violations, blocks stay small). The granularity adapts to how trustworthy the local ranking is, and the number of blocks falls out of the data with no bin count to cross-validate. To make the fit applicable to a new score, tied calibration scores are aggregated before fitting (one threshold, target the weighted average, weight the sum), redundant thresholds are trimmed after fitting, and prediction interpolates linearly between adjacent threshold values, giving a continuous non-decreasing map. A test score outside the calibration range carries no information, so I clamp the *input* to the nearest fitted score endpoint, and since these are probabilities I bound the fitted values to $[0,1]$ with $y_{\min}=0,\ y_{\max}=1$.

For the multiclass case I deliberately refuse to calibrate the joint object. A $k$-class output is a point in the $(k-1)$-dimensional simplex, and the monotonicity trick is one-dimensional — "higher score, higher probability" only means something along a single axis — so there is no order to impose in the simplex and a non-parametric fit there needs prohibitively much data as $k$ grows. Instead I reduce one-against-all: for each class $c$, the binary problem is "is the true label $c$?" against the feature "score for class $c$," and I fit an independent isotonic regression to it, yielding $r_c(x) \approx P(c \mid x)$. Fit independently, these do not sum to one. Heavy combiners exist — least squares with non-negativity, iterative log-loss coupling — and are the right tools for general code matrices, but here each one-against-all output already estimates $P(c \mid x)$ directly, so the natural reconciliation is simply to renormalize,
$$\hat p(c \mid x) = \frac{r_c(x)}{\sum_{c'} r_{c'}(x)},$$
with one numerical guard: if every $r_c(x)$ came out zero the division is $0/0$, so I floor each at a tiny $\varepsilon = 10^{-15}$ before dividing by the row sum, guaranteeing a valid distribution. The honest caveat is cost: this map is non-parametric, with as many effective degrees of freedom as there are blocks, so it needs more calibration data than the two-parameter sigmoid to avoid chasing noise — on a tiny or badly imbalanced set the sigmoid's rigidity becomes a virtue, and one judges either by strictly proper scoring rules (Brier score, negative log-likelihood) alongside reliability diagrams. For the regime here the calibration sets are large enough that the non-parametric flexibility is the right trade.

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
