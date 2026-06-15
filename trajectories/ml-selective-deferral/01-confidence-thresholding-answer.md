**Problem.** A frozen classifier must accept-or-defer each test point at a target coverage (80%). The
only editable object is the post-hoc `SelectivePolicy`; the score, the cut, and the accept test all live
in it. Errors on accepted points are costly, deferrals go to a reviewer, and the system must not
concentrate deferrals on one subgroup — but the floor ignores that last constraint.

**Key idea (the floor).** Rank by Chow's posterior-surrogate `κ(x) = max_j probs(x)_j` and cut with one
global threshold. To accept a `c`-fraction, defer the lowest-`κ` `(1 − c)`-fraction, i.e. set the cut to
the `(1 − target_coverage)` empirical quantile of `κ` on the held-out **calibration** set, then accept
`κ(x) ≥ θ` at test. Training-free, one pass, offline.

**Why it works (and where it stops).** The cut never needs `κ` to be a calibrated probability — only its
*ranking* of correct-above-incorrect, which the max class probability supplies well above chance (this is
also what the `auroc` metric reads). For a fixed ranking, a single global threshold is the *optimal*
selection rule at any coverage (admit the top-ranked prefix). That optimality holds *under the assumption
that one group-blind cut suffices* — which subgroup shift breaks: a global cut accepts more of the
high-confidence subgroup and over-defers the low-confidence one, so the `deferral_rate_gap` should be
large. The rule also ignores the calibration *labels* and *subgroup ids* the harness hands it.

**Hyperparameters.** `target_coverage = 0.80`; `quantile = clip(1 − target_coverage, 0, 1) = 0.20`;
score `np.max(probs, axis=1)`; no learned model, no per-group state.

**What to watch.** Coverage near 0.80 (a touch under, from the quantile cut); selective risk well below
full-coverage error where the base model is strong (Adult) and near-floor where it is weak (COMPAS,
Law-School); a conspicuously large deferral-rate gap — the structural ceiling that forces the next rungs.

```python
class SelectivePolicy:
    """Global confidence threshold tuned on the calibration set."""

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.threshold_: float = 0.5
        self.group_thresholds_: dict[int, float] = {}
        self.meta_model_ = None
        self.strategy_name = "confidence_thresholding"

    def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        scores = self.acceptance_score(probs, groups, X)
        quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, quantile))
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
