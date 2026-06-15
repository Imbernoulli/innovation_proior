# Softmax Response (SR) selective classification

Softmax Response equips a fixed trained classifier with a reject option by ranking inputs with
a single confidence score — the maximum softmax probability, `κ(x) = max_j softmax(x)_j` — and
accepting (predicting on) the inputs whose score clears a threshold, deferring the rest. The
threshold is set on a held-out calibration set in one of two modes: to hit a target coverage
(accept a fixed fraction) or to guarantee a target selective risk with high probability. The
classifier is never retrained; the contribution is the acceptance rule.

## Problem it solves

Given a fixed classifier `f` whose softmax outputs are available and a held-out labeled
calibration set, build a selection function `g` so the abstaining classifier `(f,g)` answers on
as many inputs as possible while keeping the error rate on the answered inputs low. Quantities:
coverage `Φ(f,g) = E[g(X)]` (fraction answered) and selective risk
`R(f,g) = E[ℓ(f(X),Y)·g(X)] / Φ(f,g)` (loss on the answered set). Trace out risk versus
coverage — the risk-coverage curve — and operate at a chosen point.

## Key idea

- **Score = maximum softmax probability.** Chow's Bayes-optimal reject rule (with known
  posteriors, 0/1 loss) thresholds `max_y P(y|x)`; the trained net's softmax max is the
  available surrogate. It is used purely as a *ranking*, so its well-documented miscalibration
  as a probability is irrelevant — the ideal score need only satisfy
  `κ(x₁) ≤ κ(x₂) ⟺ loss(x₁) ≥ loss(x₂)`. The max softmax separates correct from incorrect
  predictions well above chance, which is exactly that ranking property.
- **Selection = single global threshold.** Accept iff `κ(x) ≥ θ`. For a fixed classifier and a
  score that ranks by reliability, the optimal selection at any coverage admits the
  most-confident *prefix* of the sorted inputs — which a threshold realizes exactly — so one
  scalar `θ` is not a shortcut but the optimal selection function for that score. Sweeping `θ`
  traces the whole risk-coverage curve.
- **Threshold from a held-out calibration set.** The cut is located on data the classifier was
  not trained on (and not on the test set), so coverage/risk estimates are not optimistic.

## Two modes for setting θ

**Coverage mode (bounded-abstention).** Given target coverage `c`, accept a `c`-fraction by
rejecting the lowest-confidence `1-c`: `θ = (1-c)`-quantile of `κ` over the calibration scores.
Accept `κ(x) ≥ θ`.

**Risk mode (guaranteed risk, the SGR routine).** Given target risk `r*` and confidence `δ`,
search the sorted calibration scores for a threshold and return that threshold together with a
certified risk bound. At each candidate, compute the empirical selective risk `r̂` on the
accepted suffix and its **Clopper-Pearson** upper bound
`B*(r̂, δ/k, m)` — the largest true error `b` solving the inverse Binomial CDF

```
Σ_{j=0}^{m·r̂}  C(m, j) · b^j · (1-b)^{m-j}  =  δ/k,
```

the tightest distribution-free bound (Hoeffding etc. are looser approximations to it). Spend
`δ/k` per distinct candidate cut, `k = ⌈log₂ m⌉`, so a union bound makes the joint guarantee
`δ`.

## Guarantee (SGR theorem)

Fix threshold `g_i`; the accepted count `m_i` is random. Conditioning on `m_i = n`, the `n`
accepted points are i.i.d. from the accept-region distribution `P_{g_i}`, and the fixed-sample
bound gives `Pr{R(f|P_{g_i}) > B*(r̂_i, δ/k, n)} < δ/k`. Since
`R(f|P_{g_i}) = E[ℓ·g_i]/Φ(f,g_i) = R(f,g_i)`, the law of total probability over `n` collapses
the marginal to `δ/k`, and a union bound over the `k` distinct candidate cuts gives

```
Pr_{S_m}{ ∃ i : R(f, g_i) > B*(r̂_i, δ/k, m_i) }  ≤  δ.
```

So every distinct tested threshold's true selective risk is at most its bound w.p. `≥ 1-δ`,
hence so is the selected one. If the selected bound is `≤ r*`, the target-risk guarantee follows;
if the score is badly skewed, the returned bound may sit above `r*`, and the certificate honestly
reports that. The guarantee holds for any `κ`; the *quality* (coverage kept at a given risk)
rides on `κ` ranking well.

## Evaluating the score itself

Threshold-free, judge `κ` by the **AUROC of the acceptance score as a predictor of
correctness** = `P(κ on a correct input > κ on a wrong input)` — the probability it ranks a
true positive above a false one (chance 0.5). The risk-coverage curve and its area are the same
ranking quality read against coverage. MC-dropout uncertainty (variance of the predicted-class
response over stochastic forward passes, negated) is an alternative `κ` that ranks comparably
on some data but costs many forward passes.

## Working code

Coverage mode (the bounded-abstention selective policy):

```python
import numpy as np

TARGET_COVERAGE_DEFAULT = 0.8


class SelectivePolicy:
    """Global confidence threshold tuned on the calibration set."""

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT,
                 random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.threshold_: float = 0.5
        self.group_thresholds_: dict[int, float] = {}
        self.meta_model_ = None
        self.strategy_name = "confidence_thresholding"

    def fit(self, probs: np.ndarray, y_true: np.ndarray,
            groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        scores = self.acceptance_score(probs, groups, X)
        quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, quantile))   # (1 - coverage) quantile
        self.group_thresholds_ = {}
        self.meta_model_ = None
        return self

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray,
                         X: np.ndarray | None = None) -> np.ndarray:
        return np.max(probs, axis=1)              # kappa(x) = max_j softmax(x)_j

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray,
                       X: np.ndarray | None = None) -> np.ndarray:
        return self.acceptance_score(probs, groups, X) >= self.threshold_

    def calibration_summary(self) -> dict[str, float]:
        return {"threshold": float(self.threshold_)}
```

Risk mode (SGR: guaranteed-risk threshold via Clopper-Pearson bound + union correction):

```python
import math
import random
import numpy as np
import scipy.stats


class risk_control:
    def calculate_bound(self, delta, m, erm):
        # Invert Binom.cdf(int(m * erm), m, b) = delta.
        precision = 1e-7

        def func(b):
            return (-1 * delta) + scipy.stats.binom.cdf(int(m * erm), m, b)

        a = erm
        c = 1.0
        b = (a + c) / 2
        funcval = func(b)
        while abs(funcval) > precision:
            if a == 1.0 and c == 1.0:
                b = 1.0
                break
            elif funcval > 0:
                a = b
            else:
                c = b
            b = (a + c) / 2
            funcval = func(b)
        return b

    def bound(self, rstar, delta, kappa, residuals, split=True):
        # residuals: 0 is correct prediction, 1 is an error.
        valsize = 0.5
        probs = kappa
        FY = residuals

        if split:
            idx = list(range(len(FY)))
            random.shuffle(idx)
            slice_ = round(len(FY) * (1 - valsize))
            FY_val = FY[idx[slice_:]]
            probs_val = probs[idx[slice_:]]
            FY = FY[idx[:slice_]]
            probs = probs[idx[:slice_]]

        m = len(FY)
        probs_idx_sorted = np.argsort(probs)       # ascending confidence
        a = 0
        b = m - 1
        k = math.ceil(math.log2(m))
        deltahat = delta / k                       # union bound over k distinct cuts

        for _ in range(k + 1):
            mid = math.ceil((a + b) / 2)
            accepted = probs_idx_sorted[mid:]      # accept the most-confident suffix
            mi = len(FY[accepted])
            theta = probs[probs_idx_sorted[mid]]
            risk = sum(FY[accepted]) / mi
            if split:
                testrisk = sum(FY_val[probs_val >= theta]) / len(FY_val[probs_val >= theta])
                testcov = len(FY_val[probs_val >= theta]) / len(FY_val)
            bound = self.calculate_bound(deltahat, mi, risk)
            coverage = mi / m
            if bound > rstar:
                a = mid                            # too risky -> raise the cut
            else:
                b = mid                            # safe -> lower the cut

        return [theta, bound]
```
