A fixed classifier is already trained and I am not allowed to retrain it. In the settings I care about—an autopilot reading a scene, a model triaging a scan, a credit decision—a confident wrong answer can cost far more than no answer at all, so I want the classifier to abstain on the inputs it is least equipped to handle and hand those off to a human or slower backup, while still answering on as many of the rest as it can. The trade-off has two axes: coverage, the fraction of inputs the system actually answers on, and selective risk, the error rate measured only over the answered set. Abstaining on more inputs lowers coverage but should also lower selective risk, provided the right inputs are deferred. The object to design is therefore an acceptance rule, not a new classifier.

Existing ideas all fall short in practical settings. Chow's Bayes-optimal reject rule assumes the true class posteriors are known and thresholds the maximum posterior; a trained finite-sample model does not give us those posteriors. The version-space rejection strategy is provably optimal, but only in a noise-free, realizable setting with an enumerable hypothesis class, which does not hold for deep networks trained on noisy data. Cost-based formulations require explicit, commensurable costs for misclassification and abstention, yet in mission-critical applications those costs are themselves ill-posed. Monte-Carlo dropout offers an alternative uncertainty signal, but it costs many stochastic forward passes per input and has no built-in tie to a target coverage or risk. Even the maximum softmax probability, while correlated with correctness, is miscalibrated as an absolute probability. What is needed is a lightweight, offline rule that uses only the classifier's existing outputs and a held-out calibration set to decide which inputs to answer.

The method I propose is Softmax Response (SR) selective classification. It treats the maximum softmax probability, κ(x) = max_j softmax(x)_j, as a reliability ranking rather than a calibrated probability. The selection rule only compares inputs to each other and to a threshold, so the absolute value of κ need not mean anything; it only needs to order inputs so that more reliable predictions receive higher scores. Empirically, the maximum softmax separates correct from incorrect predictions well above chance, which is exactly the ranking property the construction needs. Moreover, for a fixed classifier and a score that ranks by reliability, a single global threshold is the optimal selection function: for any target coverage, the lowest-risk subset of that size is precisely the most-confident prefix of the sorted inputs, and a prefix is exactly what a threshold cuts out. The classifier itself is never modified; only an acceptance rule is attached downstream.

The threshold is set on a held-out calibration set that the classifier was not trained on, and never on the test set. In coverage mode, given a target coverage c, the threshold θ is the (1 − c)-quantile of the calibration scores, and an input is accepted iff κ(x) ≥ θ. This answers the highest-confidence c-fraction of inputs and defers the lowest-confidence (1 − c)-fraction. The direction matters: because κ is a confidence score, high values are accepted, so the quantile is 1 − coverage, not coverage.

In risk mode, given a target selective risk r* and confidence level δ, the sorted calibration scores are searched for the lowest cut whose true risk can be certified below r*. At each candidate threshold, the empirical risk on the accepted suffix is converted to a Clopper-Pearson upper bound by inverting the binomial cumulative distribution function, which is the tightest distribution-free bound in this setting. A union bound spends δ/k per distinct candidate cut, where k = ⌈log_2 m⌉, so the overall failure probability is at most δ. Conditioning on the random accepted count and then marginalizing over it shows the bound holds despite the random sample size. If the returned bound is at most r*, the requested risk guarantee follows; if the score ranks poorly, the certificate honestly reports that no useful cut satisfies the target.

The quality of the score itself is measured by the AUROC of κ as a predictor of correctness, the probability that a correct input receives a higher score than an incorrect one. The risk-coverage curve traces the same ranking quality against coverage. Together they confirm that thresholding the maximum softmax is a simple, computationally cheap way to add a dependable reject option to an already-deployed classifier.

```python
import math
import random
import numpy as np
import scipy.stats


TARGET_COVERAGE_DEFAULT = 0.8


class SelectivePolicy:
    """Global confidence threshold tuned on the calibration set."""

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT,
                 random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.threshold_: float = 0.5
        self.group_thresholds_: dict[int, float] = {}
        self.meta_model_ = None
        self.strategy_name = "confidence_thresholding"

    def fit(self, probs: np.ndarray, y_true: np.ndarray,
            groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        scores = self.acceptance_score(probs, groups, X)
        quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, quantile))
        self.group_thresholds_ = {}
        self.meta_model_ = None
        return self

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray,
                         X: np.ndarray | None = None) -> np.ndarray:
        return np.max(probs, axis=1)

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray,
                       X: np.ndarray | None = None) -> np.ndarray:
        return self.acceptance_score(probs, groups, X) >= self.threshold_

    def calibration_summary(self) -> dict[str, float]:
        return {"threshold": float(self.threshold_)}


class RiskControl:
    """Risk-mode threshold via Clopper-Pearson bound with a union correction."""

    def calculate_bound(self, delta: float, m: int, erm: float) -> float:
        # Invert Binom.cdf(int(m * erm), m, b) = delta.
        precision = 1e-7

        def func(b: float) -> float:
            return -delta + scipy.stats.binom.cdf(int(m * erm), m, b)

        a = erm
        c = 1.0
        b = (a + c) / 2.0
        funcval = func(b)
        while abs(funcval) > precision:
            if a == 1.0 and c == 1.0:
                b = 1.0
                break
            elif funcval > 0:
                a = b
            else:
                c = b
            b = (a + c) / 2.0
            funcval = func(b)
        return b

    def bound(self, rstar: float, delta: float, kappa: np.ndarray,
              residuals: np.ndarray, split: bool = True):
        # residuals: 0 is correct, 1 is an error.
        valsize = 0.5
        probs = np.asarray(kappa)
        FY = np.asarray(residuals)

        if split:
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(FY))
            slice_ = round(len(FY) * (1 - valsize))
            FY_val = FY[idx[slice_:]]
            probs_val = probs[idx[slice_:]]
            FY = FY[idx[:slice_]]
            probs = probs[idx[:slice_]]

        m = len(FY)
        probs_idx_sorted = np.argsort(probs)
        a = 0
        b = m - 1
        k = math.ceil(math.log2(m))
        deltahat = delta / k

        for _ in range(k + 1):
            mid = math.ceil((a + b) / 2)
            accepted = probs_idx_sorted[mid:]
            mi = len(FY[accepted])
            if mi == 0:
                a = mid
                continue
            theta = probs[probs_idx_sorted[mid]]
            risk = float(np.mean(FY[accepted]))
            bound = self.calculate_bound(deltahat, mi, risk)
            if bound > rstar:
                a = mid
            else:
                b = mid

        return theta, bound
```
