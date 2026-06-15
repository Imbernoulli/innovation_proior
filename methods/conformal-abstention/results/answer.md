# Split-conformal abstention, distilled

Split (inductive) conformal prediction can be used as a label-free abstention rule by
calibrating a threshold on the nonconformity score
`r(x) = 1 - max_j f(x)_j`. With the standard `+inf` convention at the upper endpoint, and
under exchangeability, the conformal threshold gives a finite-sample, distribution-free
statement about the marginal acceptance event:

```
P(r(X_test) <= q_hat) >= target_coverage.
```

It does not by itself certify selective accuracy or equal subgroup deferral rates; those are
separate evaluation targets.

## Problem it solves

A fixed, possibly miscalibrated classifier `f` and a downstream reviewer; design the
accept-vs-defer rule. At a target coverage `c`, accept about a `c` fraction of future
exchangeable examples and defer the rest, using only calibration-time base-model
probabilities and one fitted threshold.

## Key idea

The only assumption is **exchangeability** of the calibration scores and the test score
(i.i.d. suffices). Let `r_1,...,r_n,r_test` be exchangeable scalar nonconformity scores with
no ties, and write `r_(1) < ... < r_(n)` for the sorted calibration scores. The rank of
`r_test` among all `n+1` scores is uniform on `{1,...,n+1}`, so for any `k in {1,...,n}`:

```
P(r_test <= r_(k)) = k / (n+1).
```

Choose

```
k = ceil((n+1)(1-alpha)).
```

When `k <= n`,

```
P(r_test <= r_(k)) = ceil((n+1)(1-alpha)) / (n+1) >= 1 - alpha.
```

The `+1` is the test point: the rank is among calibration plus test, not among only the `n`
calibration points. The ceiling is forced by discreteness: rounding down can put
`k/(n+1)` below `1-alpha`. With continuous scores, the same uniform-rank fact also gives the
upper side

```
1 - alpha <= P(r_test <= q_hat) <= 1 - alpha + 1/(n+1).
```

For the conditional-on-calibration acceptance probability, let
`l = floor((n+1)alpha)`. Away from the endpoint where the ideal threshold is `+inf`, that
random coverage is

```
Beta(n + 1 - l, l),
```

so the realized operating point concentrates around `1-alpha` at the usual root-`n` scale.

## Reduction to abstention

For prediction sets, a common classification nonconformity is `1 - f(x)_y`, which uses the
true label. An abstention decision cannot use `y_test`, so the deployable score is the same
softmax idea read at the model's own top prediction:

```
s_conf(x) = max_j f(x)_j
r(x)      = 1 - s_conf(x).
```

Identify the conformal level with the target acceptance rate:

```
alpha = 1 - target_coverage.
```

Calibrate `q_hat` from the calibration nonconformities and accept exactly when
`r(x) <= q_hat`, equivalently when

```
max_j f(x)_j >= 1 - q_hat.
```

The stored threshold is therefore the confidence-space cutoff `threshold = 1 - q_hat`.

## Final algorithm

```
Input: calibration probs (n x K), target_coverage c.
1. s_i   = max_j probs[i, j]             # top-prediction confidence
2. r_i   = 1 - s_i                       # nonconformity
3. alpha = clip(1 - c, 0, 1)
4. rank  = ceil((n+1)(1-alpha)) - 1      # 0-indexed conformal rank
5. rank  = clip(rank, 0, n-1)            # finite-array guard used by the harness
6. q_hat = sort(r)[rank]
7. threshold = 1 - q_hat                 # confidence-space cutoff
Deploy: accept x iff max_j probs(x)_j >= threshold; otherwise defer.
```

Mathematically, if `ceil((n+1)(1-alpha)) = n+1`, the ideal conformal threshold is `+inf`.
The deployed implementation cannot index `r_(n+1)`, so it clamps to the largest calibration
nonconformity; ordinary target coverages below `n/(n+1)` do not hit this endpoint.

## Working code

Fills the `SelectivePolicy` slot of the offline harness:

```python
import numpy as np

TARGET_COVERAGE_DEFAULT = 0.8


class SelectivePolicy:
    """Conformal abstention using a held-out calibration set."""

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.threshold_: float = 0.5
        self.group_thresholds_: dict[int, float] = {}
        self.meta_model_ = None
        self.strategy_name = "conformal_abstention"

    def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        scores = np.max(probs, axis=1)
        nonconformity = 1.0 - scores
        n = len(nonconformity)
        alpha = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
        rank = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
        rank = int(np.clip(rank, 0, n - 1))
        self.threshold_ = float(1.0 - np.sort(nonconformity)[rank])
        self.group_thresholds_ = {}
        self.meta_model_ = None
        return self

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        return np.max(probs, axis=1)

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        return self.acceptance_score(probs, groups, X) >= self.threshold_

    def calibration_summary(self) -> dict[str, float]:
        return {"threshold": float(self.threshold_)}
```

## Relation to prior methods

- **Confidence / softmax-response thresholding**: same `max softmax` ranking signal, but a raw
  empirical cutoff lacks the conformal `n+1` rank accounting. The conformal threshold is the
  corrected order statistic.
- **Full conformal prediction**: avoids data splitting by evaluating candidate labels with
  refits or augmented fits; split conformal uses a held-out calibration set and one fitted
  base model.
- **Group-balanced calibration**: can calibrate separate thresholds within groups when the
  goal is a group-conditional accept/defer rate, at the cost of needing group labels and
  enough calibration data per group. The single pooled rule above gives a marginal guarantee.
