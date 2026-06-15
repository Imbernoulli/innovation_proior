**Problem (from step 3).** Every earlier rung used a *single decision boundary across subgroups* — the
floor and conformal on `max(probs)`, the learned gate on `P(correct)`. The learned gate even *worsened*
the deferral-rate gap (Adult 0.239 → 0.273, Law-School 0.237 → 0.324), because re-ranking under a shared
boundary sharpens which subgroup gets singled out. Improving the *score* cannot fix the gap; the boundary
must differ by group.

**Key idea.** Per-subgroup thresholds. Keep the score `s = max(probs)` (it ranks best and preserves
AUROC). For each subgroup `g`, set its cut to the `(1 − target_coverage)` quantile of `s` over the
calibration points *in that subgroup*, so a `target_coverage` fraction of each subgroup clears its own
cut — equal coverage per group, deferral rate `1 − target_coverage` everywhere. Accept iff `s(x) ≥
threshold_group(g)`, falling back to the global quantile for any subgroup unseen in calibration.

**Why it works.** Differentiating selective accuracy gives the monotonicity test `f(−τ)/F(−τ) ≥
f(τ)/(1 − F(τ))`; for roughly symmetric margins it says a group's selective accuracy *rises* with the
threshold iff its full-coverage accuracy is above 50%. Under subgroup shift the worst group sits low and
its confidences are systematically lower, so one shared cut over-defers it — the gap. Framing predict-vs-
defer as meta-classification, the equalized-odds reference (group-agnostic acceptance inside the correct
and incorrect pools) needs labels at decision time; its implementable analogue is *equal coverage per
group*, which the per-group quantile delivers exactly. Re-thresholding is a monotone group-local shift,
so the global ranking — hence AUROC — is untouched.

**Scope (honest).** Equal *coverage* ≠ equal *selective risk*: a genuinely hard worst subgroup stays
hard (frozen base model), so worst-group risk is largely unmoved and average selective risk may tick up
a hair — a deliberate accuracy-for-fairness trade. Closing the full-coverage accuracy gap is a
training-time problem this post-hoc rule cannot reach.

**Hyperparameters.** Score `np.max(probs, axis=1)`; per-group `quantile = clip(1 − target_coverage, 0,
1) = 0.20`; global-quantile fallback for unseen groups; no learned model.

**What to watch.** Deferral-rate gaps collapse (Law-School → ~0.05, Adult → ~0.16, COMPAS → ~0.07); AUROC
identical to the floor (0.853 / 0.630 / 0.614); selective / worst-group risk near the floor, perhaps a
touch worse; coverage near 0.80.

```python
class SelectivePolicy:
    """Subgroup-specific thresholds tuned on calibration data."""

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.threshold_: float = 0.5
        self.group_thresholds_: dict[int, float] = {}
        self.meta_model_ = None
        self.strategy_name = "groupwise_thresholding"

    def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        scores = self.acceptance_score(probs, groups, X)
        quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, quantile))
        self.group_thresholds_ = {}
        for group_id in np.unique(groups):
            mask = groups == group_id
            if not np.any(mask):
                continue
            self.group_thresholds_[int(group_id)] = float(np.quantile(scores[mask], quantile))
        self.meta_model_ = None
        return self

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        return np.max(probs, axis=1)

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        scores = self.acceptance_score(probs, groups, X)
        thresholds = np.asarray([self.group_thresholds_.get(int(group), self.threshold_) for group in groups], dtype=float)
        return scores >= thresholds

    def calibration_summary(self) -> dict[str, float]:
        summary = {"threshold": float(self.threshold_)}
        for group_id, threshold in self.group_thresholds_.items():
            summary[f"threshold_group_{group_id}"] = float(threshold)
        return summary
```
