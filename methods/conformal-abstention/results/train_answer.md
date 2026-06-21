We need an acceptance rule for a fixed, already-trained classifier in a high-stakes tabular setting. The classifier emits softmax scores, and a reviewer can take over the cases we defer. The goal is to accept roughly a target fraction c of future examples while keeping the rule label-free and easy to fit on a modest held-out calibration set. The obvious approach is Chow's reject option: accept when the top softmax score is above a threshold. This works well when the scores are exact posteriors, but real model scores can be miscalibrated or overconfident. Geifman and El-Yaniv's softmax-response thresholding makes the rule practical for neural nets, yet the cutoff is just an empirical quantile of a heuristic score. It gives no finite-sample, distribution-free guarantee that a fresh exchangeable test point will clear the chosen threshold at the desired rate, and it says nothing rigorous about selective risk or subgroup balance.

The limitation is not the ranking signal itself; max softmax can still put easy examples above hard ones. The missing piece is calibration of the cutoff. We need a way to place the threshold so that the acceptance event has an exact finite-sample probability statement, without modeling the score distribution. The tool for that is exchangeability. If the calibration scores and a test score are exchangeable, the rank of the test score among all n+1 pooled scores is uniform on {1,...,n+1}. This turns a statement about an unknown distribution into pure combinatorics: the probability that the test nonconformity score falls at or below the k-th sorted calibration score is exactly k/(n+1).

The method I propose is split-conformal abstention. It uses split, or inductive, conformal prediction as a label-free abstention rule. We compute a scalar nonconformity score on every example, choose a conformal rank based on the target acceptance rate, and accept exactly when the test score is no larger than the calibrated order statistic. The nonconformity score must be computable without the true label at deployment time, so we take the model's own top-prediction confidence and flip it: r(x) = 1 - max_j f(x)_j. A high max softmax means a low nonconformity, which is what we want to accept. Let c be the target coverage and alpha = 1 - c. The conformal rank is k = ceil((n+1)(1-alpha)), converted to zero-indexed coordinates and clamped to [0, n-1] for the finite-array endpoint. The threshold stored in confidence space is 1 - q_hat, where q_hat is the k-th sorted nonconformity on calibration. Deployment is then a single comparison: accept x iff max_j f(x)_j >= threshold.

The n+1 denominator is the test point counted as the unseen (n+1)-st exchangeable member. Using n in the denominator would under-cover for small calibration sets, and rounding down the rank can also break the lower bound. With continuous scores the achieved coverage is bounded by 1 - alpha <= P(r_test <= q_hat) <= 1 - alpha + 1/(n+1), so the correction is tight. The promise is marginal over the exchangeable draw; it does not certify selective risk or equal subgroup deferral rates without additional group-conditional calibration. But for the core task of controlling the accept rate of a fixed classifier on fresh data, split-conformal abstention gives a rigorous finite-sample certificate where raw softmax thresholding only offers a heuristic operating point.

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
