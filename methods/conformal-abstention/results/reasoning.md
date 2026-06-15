Let me start from the deployment problem, because that is where the usual shortcut gets too loose. I have a fixed classifier I am not allowed to retrain. For each input it gives me a vector of softmax-like scores f(x), and I have a reviewer who can take over the cases I do not want the model to answer. The obvious rule is Chow's reject option: look at max_j f(x)_j, accept when it is above a threshold, defer otherwise. If those scores were the true posteriors, that is exactly the right shape of rule for an error-reject tradeoff. But my scores are only model outputs. They can be overconfident, miscalibrated, or just badly ranked. El-Yaniv and Wiener give me the selective-classification language, and Geifman and El-Yaniv make the softmax-response threshold practical for deep nets, but the cutoff itself is still an empirical choice on a heuristic score. I can measure selective risk after the fact; I do not get a finite-sample statement that a fresh exchangeable point will land on the accept side of the cutoff at the target rate.

So I should separate two things that are easy to blur. One thing is ranking quality: does max softmax actually put easy examples above hard examples? That affects selective risk and usefulness. A different thing is calibration of the cutoff: after I choose a scalar score, can I place the threshold with an exact finite-sample accounting of how often a new point clears it? The second question might be answerable even when the first score is imperfect. I do not need the model scores to be probabilities for that; I need a symmetry argument.

Suppose the calibration points and the test point are exchangeable. I take any scalar score r on an example, compute r_1, ..., r_n on calibration and r_test on the new point, and assume ties have been broken so the ranks are well-defined. Because the joint law is unchanged by permuting the n+1 examples, the test score is equally likely to occupy any rank among the n+1 pooled scores. Conditioning on the unordered set of score values makes it completely concrete: for each target rank j there are n! assignments that put r_test in rank j out of (n+1)! equally likely assignments, so

  P(rank(r_test) = j) = 1/(n+1).

If r_(1) < ... < r_(n) are the sorted calibration scores, then r_test <= r_(k) exactly when the pooled rank of r_test is at most k. There are k such ranks, so

  P(r_test <= r_(k)) = k/(n+1).

That is the whole finite-sample engine. It does not know the distribution of r. It does not know how the classifier was trained. It only knows that the calibration scores and the test score are exchangeable.

Now I can choose the rank. I want at least a 1 - alpha probability that the new nonconformity score is no larger than the calibration threshold q_hat. If q_hat = r_(k), the exact probability is k/(n+1), so the smallest safe integer is

  k = ceil((n+1)(1 - alpha)).

Then

  P(r_test <= r_(k)) = ceil((n+1)(1 - alpha))/(n+1) >= 1 - alpha.

The +1 is not a cosmetic correction. The test point is the unseen (n+1)-st member of the exchangeable bag, so the denominator is n+1, not n. If I take the naive calibration-only rank ceil(n(1 - alpha)), the actual rank probability is ceil(n(1 - alpha))/(n+1), and for small n that can fall below 1 - alpha. The ceiling is just as forced: the rank is discrete, and the first integer whose rank probability clears the bar is ceil((n+1)(1 - alpha)); rounding down breaks the lower bound.

There is an endpoint that I should keep separate from the usual case. If alpha is so small that ceil((n+1)(1 - alpha)) = n+1, the mathematical threshold is q_hat = +infinity, because n calibration scores cannot supply an (n+1)-st order statistic. That is the ideal accept-everything endpoint. An array implementation cannot index r_(n+1), so the practical guard is to clamp the zero-indexed rank into [0, n-1]. For ordinary target coverages with ceil((n+1)(1 - alpha)) <= n, the clamp does nothing; at the endpoint it uses the largest observed calibration nonconformity as the finite fallback. The lower endpoint is analogous: if alpha is clipped to 1, the computed zero-indexed rank would be -1, so the implementation clamps it to 0.

I also want to know whether this is wastefully conservative. Under continuous scores, no ties, the rank is exactly uniform, so the achieved probability is exactly ceil((n+1)(1 - alpha))/(n+1). Since ceil(z) < z + 1,

  P(r_test <= q_hat) <= 1 - alpha + 1/(n+1).

Together with the lower bound,

  1 - alpha <= P(r_test <= q_hat) <= 1 - alpha + 1/(n+1).

So the rank correction does not push me far above the target; the slack is one rank out of n+1.

The marginal statement is averaged over the calibration draw and the test point. If I freeze one calibration set and ask what probability an infinite stream of future test points has of clearing this particular q_hat, that probability is random because q_hat is random. Reduce the calibration scores through their CDF to uniforms. With l = floor((n+1)alpha), the chosen order statistic has the distribution

  Beta(n + 1 - l, l),

away from the +infinity endpoint. Its mean is (n + 1 - l)/(n + 1), approximately 1 - alpha, and its spread shrinks at the root-n scale. That tells me what finite calibration size buys: a small calibration set gives a noisy realized operating point, while hundreds or thousands of calibration scores make the realized acceptance rate concentrate tightly around the target.

This is also where the subgroup issue shows up. A single pooled threshold controls the marginal event. It does not force each group to clear the same accept/defer rate, and it does not force the selective risk inside each group to match. If I wanted a group-conditional accept/defer rate, I would need to calibrate within groups and pay for it with smaller per-group calibration samples. For the simple pooled rule, the honest promise is marginal.

Now I have to choose the score for abstention. The set-prediction classification score 1 - f(x)_y is natural when the true label y is available for calibration and the output is a set of labels: it asks whether the true label has enough model mass. But my deploy-time decision cannot use y_test. The policy has to decide before the reviewer reveals anything. What I do have is the label the model would actually output, y_hat(x) = argmax_j f(x)_j, and the score attached to that label is max_j f(x)_j. If the model is confident in its own top prediction, I want to accept; if that confidence is low, I want to defer. So the deployable nonconformity score is

  r(x) = 1 - max_j f(x)_j.

This is label-free and computable on calibration and test in exactly the same way, which preserves the exchangeability of the scalar scores. It also makes the limitation clear: the conformal order statistic calibrates the accept/defer event for this score. It does not prove the top prediction is correct. Correctness depends on whether max softmax is a useful ranking signal for errors, which I can evaluate by selective risk or AUROC, but the finite-sample rank statement is about r_test <= q_hat.

The interface's target coverage is the fraction I want to accept. Let c = target_coverage, so alpha = 1 - c. I compute calibration confidences s_i = max_j f(x_i)_j, nonconformities r_i = 1 - s_i, sort the r_i, and take the conformal rank

  rank = ceil((n+1)(1 - alpha)) - 1

in zero-indexed coordinates, clamped to [0, n-1] for the endpoint guards. If q_hat = sort(r)[rank], the accept condition is

  r(x_test) <= q_hat.

But the harness wants a score where larger means more acceptable, so I store the equivalent confidence threshold:

  threshold = 1 - q_hat.

Then

  r(x_test) <= q_hat
  iff 1 - max_j f(x_test)_j <= q_hat
  iff max_j f(x_test)_j >= 1 - q_hat
  iff acceptance_score(x_test) >= threshold.

That is the whole reduction: the conformal score is 1 - max softmax, the threshold is the conformal order statistic in nonconformity space, and the deployed comparison is the same cutoff converted back to confidence space. The calibration labels and group ids can be passed by the harness, but this particular pooled rule does not need them.

Now the code is just the arithmetic above in the harness's policy shape:

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

The causal chain is now tight: softmax thresholding gives the right practical ranking shape but no finite-sample accounting for a fresh exchangeable point; exchangeability turns the unknown score distribution into a uniform rank; the rank forces ceil((n+1)(1 - alpha)) and the zero-indexed clamp in code; alpha is 1 - target_coverage; the abstention score is the deployable nonconformity 1 - max softmax; storing 1 - q_hat makes the runtime rule a single comparison on max softmax.
