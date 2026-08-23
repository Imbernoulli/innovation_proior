class SelectivePolicy:
    """Budget-faithful selection: land on the coverage target, then cut risk.

    Variant objective: the cutoff estimated on calibration data must
    reproduce the 80% acceptance rate on the test split almost exactly,
    and the accepted pool should carry the least error achievable at that
    realised budget. Off-budget operation forfeits the variant's claim,
    however good the other columns look.

    Structural hooks provided by this scaffold:
      - conformal-style order statistic: ``fit`` places the cutoff with a
        finite-sample (n+1) adjustment instead of a plain quantile, so the
        calibration-to-test coverage transfer is unbiased.
      - ``tie_break``: weight of a deterministic secondary key added to the
        margin score; heavy score ties make realised coverage jump in
        chunks, which is exactly the transfer failure to engineer away.
      - ``cal_coverage_`` / ``coverage_slack_``: the acceptance rate the
        fitted cutoff realises on calibration data and its distance to the
        budget — the audit trail behind the coverage claim.
    """

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0,
                 tie_break: float = 1e-6):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.tie_break = float(tie_break)
        self.threshold_: float = 0.5
        self.cal_coverage_: float = float("nan")
        self.coverage_slack_: float = float("nan")
        self.meta_model_ = None
        self.strategy_name = "finite_sample_budget_quantile"

    def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        scores = self.acceptance_score(probs, groups, X)
        n = len(scores)
        # Conformal-style placement: defer the ceil((n+1)*(1-c)) - 1 lowest
        # calibration scores so the expected test coverage matches the target.
        n_defer = int(np.ceil((n + 1) * (1.0 - self.target_coverage))) - 1
        n_defer = int(np.clip(n_defer, 0, max(n - 1, 0)))
        order = np.sort(scores)
        if n_defer > 0:
            self.threshold_ = float(order[n_defer])
        else:
            self.threshold_ = float(order[0] - 1.0)
        accept = scores >= self.threshold_
        self.cal_coverage_ = float(accept.mean())
        self.coverage_slack_ = float(self.cal_coverage_ - self.target_coverage)
        return self

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        p0 = probs[:, 0].astype(float)
        p1 = probs[:, 1].astype(float)
        # Margin is the primary key; the positive-class probability acts as a
        # deterministic tie-breaker so realised coverage moves smoothly.
        return np.abs(p1 - p0) + self.tie_break * p1

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        scores = self.acceptance_score(probs, groups, X)
        return scores >= self.threshold_

    def calibration_summary(self) -> dict[str, float]:
        return {
            "threshold": float(self.threshold_),
            "cal_coverage": float(self.cal_coverage_),
            "coverage_slack": float(self.coverage_slack_),
        }
