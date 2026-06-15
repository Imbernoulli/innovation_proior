**Problem (from step 2).** Conformal fixed the *cut* (COMPAS coverage 0.7754 → 0.7810) but left the
*score* as raw `max(probs)`, a weak correctness-ranker where the base model is hard (COMPAS AUROC 0.630,
Law-School 0.614). The next gain is a better ordering — but the base classifier is frozen, so jointly
training a reject head is off the table; only a *post-hoc* learned gate on the model's outputs is allowed.

**Key idea.** Recast acceptance as its own supervised problem. On calibration, define the binary target
`correct_i = 1[argmax(probs_i) = y_i]` (the harness hands the labels, which earlier rungs ignored) and
train a compact meta-classifier to predict `P(correct | features)` from quantities computable at test
time — `_confidence_features` = `[p1, max_prob, margin, entropy, group, X[:,0]]`. Use that predicted
correctness as the acceptance score; cut at the `(1 − target_coverage)` calibration quantile of it.

**Why it works.** `max(probs)` is one hand-chosen correctness feature; a learned weighting over several
(entropy reads distribution *shape*, group/`X[:,0]` let confidence mean different things in different
regions) can rank better exactly where the bare max-prob is weak, while reproducing the floor where it is
already strong (Adult). Logistic regression keeps it convex, offline, millisecond-fast; `class_weight=
"balanced"` is essential — `correct` is imbalanced (model accuracy ≫ 50%), and without it the gate
collapses to "accept everything" and stops ranking. The cut stays the plain quantile (not conformal):
the learned score is fit on the calibration data, so its exchangeability with a fresh point is no longer
clean.

**Why not group-aware.** The meta-classifier predicts correctness *marginally* and sees `group` as just
one feature; it has no per-group coverage objective, so it can — and likely will — concentrate deferrals
on the least-likely-correct subgroup, possibly *worse* than the floor. That tension is left for the cut.

**Hyperparameters.** `LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced",
random_state=seed)` on `StandardScaler`-ed `_confidence_features`; cut `quantile = clip(1 −
target_coverage, 0, 1) = 0.20`; fallback to `max(probs)` if the meta-model is unset.

**What to watch.** AUROC / selective risk room to drop on COMPAS, Law-School; Adult ≈ floor; deferral-rate
gaps unimproved or *worse* (no per-group objective) — the explicit motivation for the next rung.

```python
class SelectivePolicy:
    """Compact learned gate that predicts correctness from confidence features."""

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.threshold_: float = 0.5
        self.group_thresholds_: dict[int, float] = {}
        self.meta_model_ = None
        self.strategy_name = "learned_deferral"

    def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        features = _confidence_features(probs, groups, X)
        correct = (np.argmax(probs, axis=1) == y_true).astype(int)
        self.meta_model_ = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.meta_model_.fit(features, correct)
        scores = self.meta_model_.predict_proba(features)[:, 1]
        quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, quantile))
        self.group_thresholds_ = {}
        return self

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        if self.meta_model_ is None:
            return np.max(probs, axis=1)
        features = _confidence_features(probs, groups, X)
        return self.meta_model_.predict_proba(features)[:, 1]

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        return self.acceptance_score(probs, groups, X) >= self.threshold_

    def calibration_summary(self) -> dict[str, float]:
        return {"threshold": float(self.threshold_)}
```
