Let me start from the deployment problem, because that is where the usual shortcut gets too loose. I have a fixed classifier I am not allowed to retrain. For each input it gives me a vector of softmax-like scores f(x), and I have a reviewer who can take over the cases I do not want the model to answer. The obvious rule is Chow's reject option: look at max_j f(x)_j, accept when it is above a threshold, defer otherwise. If those scores were the true posteriors, that is exactly the right shape of rule for an error-reject tradeoff. But my scores are only model outputs. They can be overconfident, miscalibrated, or just badly ranked. El-Yaniv and Wiener give me the selective-classification language, and Geifman and El-Yaniv make the softmax-response threshold practical for deep nets, but the cutoff itself is still an empirical choice on a heuristic score. I can measure selective risk after the fact; I do not get a finite-sample statement that a fresh exchangeable point will land on the accept side of the cutoff at the target rate.

So I should separate two things that are easy to blur. One thing is ranking quality: does max softmax actually put easy examples above hard examples? That affects selective risk and usefulness. A different thing is calibration of the cutoff: after I choose a scalar score, can I place the threshold with an exact finite-sample accounting of how often a new point clears it? The second question might be answerable even when the first score is imperfect. I do not need the model scores to be probabilities for that; I need a symmetry argument.

Suppose the calibration points and the test point are exchangeable. I take any scalar score r on an example, compute r_1, ..., r_n on calibration and r_test on the new point, and assume ties have been broken so the ranks are well-defined. Because the joint law is unchanged by permuting the n+1 examples, the test score is equally likely to occupy any rank among the n+1 pooled scores. Conditioning on the unordered set of score values makes it completely concrete: for each target rank j there are n! assignments that put r_test in rank j out of (n+1)! equally likely assignments, so

  P(rank(r_test) = j) = 1/(n+1).

If r_(1) < ... < r_(n) are the sorted calibration scores, then r_test <= r_(k) exactly when the pooled rank of r_test is at most k. There are k such ranks, so

  P(r_test <= r_(k)) = k/(n+1).

This does not know the distribution of r. It does not know how the classifier was trained. It only knows that the calibration scores and the test score are exchangeable. Before I lean on it, let me sanity-check the counting at the smallest interesting size. Take n = 1: there is one calibration score and one test score, two pooled values, and exchangeability says r_test is below or above the single calibration value with probability 1/2 each. The formula gives P(r_test <= r_(1)) = 1/(n+1) = 1/2. That matches, and it also matches my intuition that with a single calibration point I genuinely cannot pin coverage more finely than a coin flip.

Now I can choose the rank. I want at least a 1 - alpha probability that the new nonconformity score is no larger than the calibration threshold q_hat. If q_hat = r_(k), the exact probability is k/(n+1), so the smallest safe integer is

  k = ceil((n+1)(1 - alpha)).

Then

  P(r_test <= r_(k)) = ceil((n+1)(1 - alpha))/(n+1) >= 1 - alpha.

I want to be sure the +1 inside that ceiling is actually doing work and is not just bookkeeping I could drop. The tempting alternative is to rank the test score only against the n calibration scores, the way one would set an empirical quantile of a sample, which would give k = ceil(n(1 - alpha)). Let me put real numbers on both at a small size where I can check coverage exactly. Take alpha = 0.2 (target coverage 0.8) and n = 5. The naive rank is ceil(5 * 0.8) = 4, so it would threshold at r_(4). But the probability that the test score actually falls at or below r_(4) is, by the same uniform-rank fact, 4/(n+1) = 4/6 = 0.667 — well below the 0.8 I asked for. The conformal rank is ceil(6 * 0.8) = 5, thresholding at r_(5), with probability 5/6 = 0.833 >= 0.8. So at n = 5 the naive cutoff undershoots the target by thirteen points and the conformal cutoff clears it; the denominator really is n+1, because the test point is the unseen (n+1)-st member of the bag, and the numerator has to be ceiled because the rank is discrete and the first integer whose probability clears the bar is ceil((n+1)(1 - alpha)). Rounding down, or dividing by n, breaks the lower bound at small n. (At larger n the two ranks often coincide after rounding, so the gap is easy to miss if I only look at n in the hundreds, which is exactly why the small case is the honest test.)

There is an endpoint that I should keep separate from the usual case. If alpha is so small that ceil((n+1)(1 - alpha)) = n+1, the mathematical threshold is q_hat = +infinity, because n calibration scores cannot supply an (n+1)-st order statistic. That is the ideal accept-everything endpoint. An array implementation cannot index r_(n+1), so the practical guard is to clamp the zero-indexed rank into [0, n-1]. For ordinary target coverages with ceil((n+1)(1 - alpha)) <= n, the clamp does nothing; at the endpoint it uses the largest observed calibration nonconformity as the finite fallback. The lower endpoint is analogous: if alpha is clipped to 1, the computed zero-indexed rank would be -1, so the implementation clamps it to 0.

I also want to know whether this is wastefully conservative. Under continuous scores, no ties, the rank is exactly uniform, so the achieved probability is exactly ceil((n+1)(1 - alpha))/(n+1). Since ceil(z) < z + 1,

  P(r_test <= q_hat) <= 1 - alpha + 1/(n+1).

Together with the lower bound,

  1 - alpha <= P(r_test <= q_hat) <= 1 - alpha + 1/(n+1).

Let me check that this band is not vacuous. At n = 5, alpha = 0.2 the achieved probability is 5/6 = 0.833, an excess of 0.033 over the target, and the bound 1/(n+1) = 0.167 sits comfortably above it. At n = 49 the achieved probability is 40/50 = 0.800 exactly, excess zero, well inside the 0.02 band. So the rank correction does not push me far above the target; the slack is at most one rank out of n+1, and at the operating points I care about it is often nil.

The marginal statement is averaged over the calibration draw and the test point. If I freeze one calibration set and ask what probability an infinite stream of future test points has of clearing this particular q_hat, that probability is random because q_hat is random. Reduce the calibration scores through their CDF to uniforms, so q_hat becomes the k-th order statistic of n uniform draws, with k = ceil((n+1)(1 - alpha)). The k-th order statistic of n uniforms is Beta(k, n + 1 - k). With l = floor((n+1)alpha), I have k = n + 1 - l away from the +infinity endpoint, so n + 1 - k = l, and the distribution is

  Beta(n + 1 - l, l).

I should confirm the index bookkeeping rather than trust it. Take n = 49, alpha = 0.2: then l = floor(50 * 0.2) = 10 and k = ceil(50 * 0.8) = 40, and indeed k = n + 1 - l = 50 - 10 = 40, with n + 1 - k = 10 = l, so Beta(k, n+1-k) and Beta(n+1-l, l) are the same Beta(40, 10). Its mean is (n + 1 - l)/(n + 1) = 40/50 = 0.8, exactly 1 - alpha, and its standard deviation is sqrt(40*10/(50^2 * 51)) ≈ 0.056, so the realized acceptance rate of a frozen calibration set sits within a few points of the target. That tells me what finite calibration size buys: a small calibration set gives a noisy realized operating point, while hundreds or thousands of calibration scores shrink that spread at the root-n scale and make the realized acceptance rate concentrate tightly around the target.

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

Before I commit to that sign-flip, let me run it once by hand, because confusing nonconformity space with confidence space is the easy way to invert the whole rule. Take five calibration points with top-softmax confidences 0.55, 0.70, 0.80, 0.90, 0.95, and c = 0.8 so alpha = 0.2. The nonconformities are 0.45, 0.30, 0.20, 0.10, 0.05; sorted, r = [0.05, 0.10, 0.20, 0.30, 0.45]. The zero-indexed rank is ceil(6 * 0.8) - 1 = 5 - 1 = 4, so q_hat = sort(r)[4] = 0.45 and threshold = 1 - 0.45 = 0.55. Now a test point: confidence 0.50 has nonconformity 0.50 > 0.45, so it should defer — and 0.50 >= 0.55 is False, deferred, consistent. Confidence 0.78 has nonconformity 0.22 <= 0.45, accept — and 0.78 >= 0.55 is True, accepted, consistent. Confidence 0.95 accepts both ways. So the confidence-space comparison reproduces the nonconformity-space rule exactly, and the stored threshold is on the right side of the flip.

So the rule that emerges is: the conformal score is 1 - max softmax, the threshold is the conformal order statistic in nonconformity space, and the deployed comparison is the same cutoff converted back to confidence space. The calibration labels and group ids can be passed by the harness, but this particular pooled rule does not need them.

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

Reading back over the chain: softmax thresholding gives the right practical ranking shape but no finite-sample accounting for a fresh exchangeable point; exchangeability turns the unknown score distribution into a uniform rank, which the n = 1 and n = 5 checks bear out; the rank forces ceil((n+1)(1 - alpha)) and the zero-indexed clamp in code, with the +1 demonstrably load-bearing at small n; alpha is 1 - target_coverage; the abstention score is the deployable nonconformity 1 - max softmax; and the hand-traced example confirms that storing 1 - q_hat makes the runtime rule a single comparison on max softmax.
