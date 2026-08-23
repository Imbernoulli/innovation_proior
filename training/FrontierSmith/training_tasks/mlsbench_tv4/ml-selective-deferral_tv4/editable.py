class SelectivePolicy:
    """Ranking-first selection: the acceptance score is the deliverable.

    Variant objective: the score must order test cases by their probability
    of being correctly classified — the AUROC column is the primary target
    — and the accept/defer rule is nothing more than a cut through that
    ordering at the budgeted coverage. Risk improvements are expected to
    follow from ordering quality, not from threshold gymnastics.

    Structural hooks provided by this scaffold:
      - a compact correctness meta-ranker: logistic regression over the
        fixed confidence features (positive-class probability, max
        probability, margin, entropy), trained on calibration correctness.
      - ``fallback_``: True when calibration correctness is degenerate
        (all right or all wrong) so the ranker cannot be fit and the score
        falls back to the raw margin.
      - ``meta_auroc_``: in-sample ranking quality of the fitted score on
        calibration data — the first number a stronger ranker should move.
    """

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.threshold_: float = 0.5
        self.meta_model_ = None
        self.fallback_ = True
        self.meta_auroc_: float = 0.5
        self.strategy_name = "correctness_meta_ranker"

    def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        y_true = np.asarray(y_true).astype(int)
        correct = (np.argmax(probs, axis=1) == y_true).astype(int)
        self.meta_model_ = None
        self.fallback_ = True
        if len(np.unique(correct)) == 2:
            feats = _confidence_features(probs)
            model = LogisticRegression(max_iter=1000, random_state=self.random_state)
            model.fit(feats, correct)
            self.meta_model_ = model
            self.fallback_ = False
        scores = self.acceptance_score(probs, groups, X)
        self.meta_auroc_ = float(_safe_roc_auc(correct, scores))
        quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, quantile))
        return self

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        if self.meta_model_ is not None:
            return self.meta_model_.predict_proba(_confidence_features(probs))[:, 1]
        return np.abs(probs[:, 1] - probs[:, 0]).astype(float)

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        scores = self.acceptance_score(probs, groups, X)
        return scores >= self.threshold_

    def calibration_summary(self) -> dict[str, float]:
        return {
            "threshold": float(self.threshold_),
            "meta_auroc": float(self.meta_auroc_),
            "fallback": float(self.fallback_),
        }
