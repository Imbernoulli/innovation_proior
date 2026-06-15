**Problem (from step 1).** The global confidence threshold's *ranking* was the base model's to give
(AUROC 0.853/0.630/0.614, untouched here), but its *cut* treated the calibration quantile as the
population quantile and drifted under target on COMPAS (achieved coverage 0.7754 vs 0.80). I want the
cut to carry a finite-sample guarantee that a fresh point clears it at the target rate.

**Key idea.** Split-conformal calibration. The score stays `max(probs)`; convert it to a *nonconformity*
`r(x) = 1 − max_j probs(x)_j` (label-free, computed identically on calibration and test, so it is
exchangeable). By exchangeability of the `n` calibration scores and the test score, the pooled rank of
`r_test` is uniform on `{1,…,n+1}`, so `P(r_test ≤ r_(k)) = k/(n+1)`. To accept a `(1 − α)`-fraction
(`α = 1 − target_coverage`), take the conformal rank `k = ceil((n+1)(1 − α))`, i.e. zero-indexed
`rank = ceil((n+1)(1 − α)) − 1` clamped to `[0, n−1]`, set `q_hat = sort(r)[rank]`, and store the
equivalent confidence threshold `threshold = 1 − q_hat`. Deploy `max(probs) ≥ threshold`.

**Why it works.** The `+1` denominator counts the unseen test point as the `(n+1)`-st member of the bag;
the naive `ceil(n(1 − α))` rank can fall below `1 − α` for finite `n` — that is the COMPAS under-coverage.
With no ties the achieved coverage is `1 − α ≤ P ≤ 1 − α + 1/(n+1)`: the correction fixes the unlucky
draw without over-shooting. The guarantee is on the *marginal* accept event only — a pooled threshold
does not equalize per-group coverage or per-group risk, so the deferral-rate gap is left for a later rung.

**Hyperparameters.** `α = 1 − target_coverage = 0.20`; nonconformity `1 − max(probs)`; conformal rank
`ceil((n+1)(1 − α)) − 1`, clamped; no learned model, no per-group state.

**What to watch.** COMPAS coverage should rise toward 0.80; Adult and Law-School near-identical to step 1
(cut lands on the same score); AUROC unchanged everywhere; deferral-rate gaps still large (marginal-only).

```python
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
