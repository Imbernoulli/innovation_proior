class SelectivePolicy:
    """Setting-agnostic selection: one fixed rule, judged by its weakest run.

    Variant objective: the identical policy is dropped onto three unrelated
    tabular problems with no per-dataset tuning, and the combined grade is
    bounded by whichever dataset it handles worst. The goal is a small
    cross-dataset spread on every reported column, achieved through
    distribution-free mechanics rather than knobs fitted to any one task.

    Structural hooks provided by this scaffold:
      - knob-free operation: the score is the untransformed max-probability
        and the cutoff is its budget quantile — nothing to tune per task.
      - resampling audit: ``fit`` recomputes the cutoff on two disjoint
        halves of the calibration set; ``threshold_spread_`` records how
        far the two estimates disagree, a direct instability reading.
      - ``half_thresholds_``: the per-half cutoffs behind that spread,
        kept for the stability argument the variant must make.
    """

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.threshold_: float = 0.5
        self.threshold_spread_: float = 0.0
        self.half_thresholds_: tuple[float, float] = (0.5, 0.5)
        self.meta_model_ = None
        self.strategy_name = "knob_free_stability_audit"

    def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        scores = self.acceptance_score(probs, groups, X)
        quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, quantile))
        # Stability audit: how far does the cutoff move under a split-half
        # refit? Large spread flags a rule that will not travel well.
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(len(scores))
        half = len(scores) // 2
        if half >= 1:
            t_a = float(np.quantile(scores[perm[:half]], quantile))
            t_b = float(np.quantile(scores[perm[half:]], quantile))
            self.half_thresholds_ = (t_a, t_b)
            self.threshold_spread_ = float(abs(t_a - t_b))
        return self

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        return np.max(probs, axis=1).astype(float)

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        scores = self.acceptance_score(probs, groups, X)
        return scores >= self.threshold_

    def calibration_summary(self) -> dict[str, float]:
        return {
            "threshold": float(self.threshold_),
            "threshold_spread": float(self.threshold_spread_),
            "half_threshold_low": float(min(self.half_thresholds_)),
            "half_threshold_high": float(max(self.half_thresholds_)),
        }
