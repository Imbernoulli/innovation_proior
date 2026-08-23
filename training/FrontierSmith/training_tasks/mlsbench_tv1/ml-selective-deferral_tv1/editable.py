class SelectivePolicy:
    """One globally comparable acceptance score with group-equitable deferral.

    Variant objective: at the fixed coverage budget, shrink the subgroup
    deferral-rate gap while holding worst-group accepted error AND the
    global correctness-ranking quality of the single acceptance score.

    Structural hooks provided by this scaffold:
      - ``group_offset_``: per-group additive corrections applied to the one
        base score. All zeros by default, so the placeholder is a plain
        global max-probability threshold — it exhibits exactly the burden
        imbalance this variant asks you to remove.
      - ``equity_strength``: scales how strongly ``fit`` may use calibration
        group statistics to move deferral burden (0.0 default = off).
      - ``group_stats_``: per-group calibration bookkeeping (deferral share,
        accepted error) — the evidence any burden-equalizing correction
        should be derived from.
    """

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0,
                 equity_strength: float = 0.0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.equity_strength = float(equity_strength)
        self.threshold_: float = 0.5
        self.group_offset_: dict[int, float] = {}
        self.group_stats_: dict[int, dict[str, float]] = {}
        self.meta_model_ = None
        self.strategy_name = "global_threshold_group_offsets"

    def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        groups = np.asarray(groups)
        y_true = np.asarray(y_true)
        # Placeholder keeps offsets at zero; a real method fills them from
        # group_stats_ (scaled by equity_strength) so that the SAME threshold
        # spreads deferrals evenly without scrambling the global ranking.
        self.group_offset_ = {int(g): 0.0 for g in np.unique(groups)}
        scores = self.acceptance_score(probs, groups, X)
        quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, quantile))

        # Calibration-side burden audit: who is deferred, and how reliable
        # are each group's accepted predictions under the current score.
        pred = np.argmax(probs, axis=1)
        accept = scores >= self.threshold_
        self.group_stats_ = {}
        for g in np.unique(groups):
            mask = groups == g
            acc = accept[mask]
            if acc.any():
                accepted_error = float(np.mean(pred[mask][acc] != y_true[mask][acc]))
            else:
                accepted_error = 1.0
            self.group_stats_[int(g)] = {
                "deferral_rate": float(1.0 - acc.mean()),
                "accepted_error": accepted_error,
                "n": float(mask.sum()),
            }
        return self

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        base = np.max(probs, axis=1).astype(float)
        if self.group_offset_ and self.equity_strength != 0.0:
            offsets = np.array([self.group_offset_.get(int(g), 0.0) for g in np.asarray(groups)])
            base = base + self.equity_strength * offsets
        return base

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        scores = self.acceptance_score(probs, groups, X)
        return scores >= self.threshold_

    def calibration_summary(self) -> dict[str, float]:
        summary = {"threshold": float(self.threshold_)}
        for g, stats in self.group_stats_.items():
            summary[f"group{g}_deferral_rate"] = stats["deferral_rate"]
            summary[f"group{g}_accepted_error"] = stats["accepted_error"]
        return summary
