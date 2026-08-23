class SelectivePolicy:
    """Asymmetric-cost gate: a wrong accept is priced far above a deferral.

    Variant objective: treat every accepted mistake as roughly five times
    as expensive as sending the case to review. The reported error columns
    (overall and worst-subgroup accepted error) are the dominant cost
    terms; the budget is respected but never used as an excuse to wave
    through decisions whose expected cost exceeds the price of a review.

    Structural hooks provided by this scaffold:
      - ``wrong_accept_cost``: the cost ratio (a review costs 1.0). Its
        indifference point ``1 - 1/cost`` is the confidence below which
        acceptance is irrational no matter what the budget says.
      - ``cost_floor_``: that indifference confidence, materialised at fit
        time; the operating cutoff is the harder of the budget cutoff and
        this floor.
      - ``floor_binds_``: fraction of calibration cases where the floor,
        not the budget, decides — evidence for the asymmetry argument.
    """

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0,
                 wrong_accept_cost: float = 5.0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.wrong_accept_cost = float(max(wrong_accept_cost, 1.0 + 1e-9))
        self.threshold_: float = 0.5
        self.cost_floor_: float = 0.0
        self.floor_binds_: float = 0.0
        self.meta_model_ = None
        self.strategy_name = "expected_cost_floor"

    def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        scores = self.acceptance_score(probs, groups, X)
        quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
        budget_cut = float(np.quantile(scores, quantile))
        # Accepting pays (1 - confidence) * cost in expectation; deferring
        # pays 1. Below the indifference confidence, deferral is cheaper.
        self.cost_floor_ = float(1.0 - 1.0 / self.wrong_accept_cost)
        self.threshold_ = float(max(budget_cut, self.cost_floor_))
        self.floor_binds_ = float(np.mean((scores >= budget_cut) & (scores < self.threshold_)))
        return self

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        return np.max(probs, axis=1).astype(float)

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        scores = self.acceptance_score(probs, groups, X)
        return scores >= self.threshold_

    def calibration_summary(self) -> dict[str, float]:
        return {
            "threshold": float(self.threshold_),
            "cost_floor": float(self.cost_floor_),
            "floor_binds": float(self.floor_binds_),
        }
