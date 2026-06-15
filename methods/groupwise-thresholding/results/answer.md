# Groupwise thresholding, distilled

Groupwise thresholding is a confidence-based selective-prediction (accept/defer) rule that gives
*each subgroup its own acceptance threshold*, calibrated so that every subgroup is covered at the
same target rate. It ranks points by softmax response (the maximum class probability) and sets,
per subgroup, the threshold at the `(1 − target_coverage)` quantile of that score *within* the
subgroup. A point is accepted iff its score clears its own subgroup's threshold. This equalizes
coverage — and therefore the deferral rate — across subgroups while leaving the underlying
softmax-response score unchanged.

## Problem it solves

A base classifier and its train/calibration/test pipeline are fixed. The only design choice is
the acceptance rule: at a target coverage (e.g. predict on 80% of test points, defer the rest to
a human), decide which points to accept. Under subgroup shift a single *global* confidence
threshold defers disproportionately on the harder subgroup and can move its selective accuracy
the wrong way. The rule must, at the target coverage, keep selective risk and worst-subgroup
selective risk low, keep the deferral-rate gap across subgroups small, preserve the acceptance
score's AUROC for correctness, and fit offline on calibration data without retraining the base
model.

## Key idea

1. **Acceptance score = softmax response.** `s(x) = max_y P̂(y|x)`, the maximum class
   probability. It is a monotone transform of the logit confidence
   `ĉ(x) = ½ log(P̂(ŷ|x)/(1−P̂(ŷ|x)))` (plus `½ log(k−1)` for `k` classes), so it ranks points
   identically and yields the same accuracy-coverage curve and the same AUROC. Thresholding `s`
   is Chow's optimal reject rule for a fixed cost.

2. **Why one global threshold fails.** Summarize the classifier by its margin CDF `F`
   (`m(x) = ĉ(x)` if correct, `−ĉ(x)` if wrong). Selective accuracy is
   `A_F(τ) = (1−F(τ))/(F(−τ)+1−F(τ))`. Differentiating, `A_F` increases at `τ` iff
   `f(−τ)/F(−τ) ≥ f(τ)/(1−F(τ))`. For a symmetric **left-log-concave** margin distribution (CDF
   log-concave on `(−∞, μ]`; satisfied even by bimodal Gaussian mixtures), `A_F(τ)` is monotone
   increasing if full-coverage accuracy `A_F(0) ≥ ½` and monotone **decreasing** if `A_F(0) ≤ ½`.
   The worst subgroup, sitting below 50% on spurious-correlation data, therefore gets *worse* as
   the global threshold rises, while the above-50% average improves — and the worst subgroup is
   also covered less and deferred more.

3. **The group-agnostic reference and why a shared boundary can't reach it.** Frame predict-vs-
   defer as a meta-task (TP = predict-and-correct, FP = predict-and-wrong). A reference that keeps
   the same total correct/incorrect counts but spreads them across subgroups by their
   full-coverage shares — `C̃_g(τ) = C_g(0)·C(τ)/C(0)`, `Ĩ_g(τ) = I_g(0)·I(τ)/I(0)` — gives every
   subgroup the same TPR and FPR, so it satisfies equalized odds for the accept/defer decision. A
   single shared threshold can match this on the worst subgroup only if
   `f_bg(0)/f_wg(0) ≤ (1−A_bg(0))/(1−A_wg(0))` (ratio of full-coverage errors) — harder the larger
   the disparity — and for translated log-concave subgroups it underperforms the reference at
   *every* threshold.

4. **The rule: per-subgroup thresholds at equal coverage.** Use the available subgroup ids to
   enforce the deferral-rate objective directly: give each subgroup its own threshold at the same
   within-subgroup
   `(1 − target_coverage)` quantile of `s`. Every subgroup then keeps coverage = `target_coverage`
   ⇒ equal deferral rate (small deferral-rate gap), with no labels needed at decision time.
   Re-thresholding does not alter the returned SR score, so the score AUROC is preserved. Fall
   back to the global quantile for any subgroup unseen in calibration. This is an equal-coverage
   post-processing rule, not the label-using equalized-odds reference itself.

5. **Honest scope.** Equalizing *coverage* is not equalizing *accuracy*: a genuinely hard subgroup
   is not repaired just by moving thresholds on the same score. Closing the full-coverage accuracy
   gap is a training-time problem (e.g. group DRO); this rule, bound to a fixed base model, removes
   the deferral-rate disparity and preserves ranking quality.

## Final algorithm

```
Inputs: target coverage c; calibration probs P (n × n_classes), subgroup ids g (n).
Score:  s_i = max_k P[i, k]                       # softmax response
Fit:    q = 1 − c                                  # quantile
        threshold_global = quantile(s, q)          # fallback
        for each subgroup id u in unique(g):
            threshold[u] = quantile({ s_i : g_i = u }, q)
Decide: accept point x with score s and subgroup u  iff  s ≥ threshold.get(u, threshold_global)
```

At calibration, each subgroup has a `c` fraction of its own points above its own threshold, so
every subgroup is covered at rate `c` and deferred at rate `1 − c`.

## Working code

Filling the `SelectivePolicy` slot of the fixed acceptance-rule harness:

```python
import numpy as np

TARGET_COVERAGE_DEFAULT = 0.8


class SelectivePolicy:
    """Subgroup-specific thresholds tuned on calibration data."""

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.threshold_: float = 0.5
        self.group_thresholds_: dict[int, float] = {}
        self.meta_model_ = None
        self.strategy_name = "groupwise_thresholding"

    def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray,
            X: np.ndarray | None = None) -> "SelectivePolicy":
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

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray,
                         X: np.ndarray | None = None) -> np.ndarray:
        return np.max(probs, axis=1)

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray,
                       X: np.ndarray | None = None) -> np.ndarray:
        scores = self.acceptance_score(probs, groups, X)
        thresholds = np.asarray(
            [self.group_thresholds_.get(int(g), self.threshold_) for g in groups], dtype=float)
        return scores >= thresholds

    def calibration_summary(self) -> dict[str, float]:
        summary = {"threshold": float(self.threshold_)}
        for group_id, threshold in self.group_thresholds_.items():
            summary[f"threshold_group_{group_id}"] = float(threshold)
        return summary
```

## Relation to prior methods

- **Global confidence thresholding (Chow / softmax response):** the special case of one shared
  threshold for all subgroups. Provably over-defers the left-shifted worst subgroup and can lower
  its selective accuracy; can fail to match the group-agnostic reference when the disparity is
  large.
- **Split-conformal abstention:** adds a marginal finite-sample coverage guarantee to the single
  threshold, but the guarantee is over pooled data, so it still does not equalize coverage across
  subgroups.
- **Learned deferral:** fits a correctness predictor; inherits the underlying score's
  subgroup-dependent miscalibration and gives no direct handle on per-subgroup deferral rate.
- **Group-agnostic equalized-odds reference:** the label-using construction that keeps the same
  total correct/incorrect predictions while redistributing them without regard to group identity.
  Per-subgroup quantile thresholds do not reproduce that construction; they implement the
  available equal-coverage objective directly.
- **Group DRO (training-time sibling):** closes full-coverage accuracy gaps by retraining with
  group labels; complementary, and out of scope when the base model is fixed.
