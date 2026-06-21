The canonical result I will describe is the Neyman-Pearson lemma. It is the central answer to the question of how to test one simple statistical hypothesis against another simple alternative when the two error probabilities are in tension. Suppose I observe a single random draw, or a vector of draws, whose distribution is either P0 under the null hypothesis H0 or P1 under the alternative hypothesis H1. A test is simply a rule that tells me, for each possible observation, how often I should reject H0. If the rule rejects with high probability when H1 is true, it has high power; if it rejects with low probability when H0 is true, it has small size. The Neyman-Pearson lemma tells me how to spend a fixed size budget alpha in the place where it buys the most power.

The key insight is that I should not look for observations that are merely rare under H0. An observation can be rare under H0 and also rare under H1, in which case rejecting there does almost nothing to help me detect the alternative. What matters is the exchange rate: how much probability under H1 do I gain for each unit of probability under H0 that I spend? That exchange rate is the likelihood ratio p1(x)/p0(x), where p0 and p1 are densities or probability mass functions with respect to a common dominating measure. Points with a high likelihood ratio give me a lot of power per unit of size, so they should enter the rejection region first. Points with a low likelihood ratio are expensive in terms of size and give little reward, so they should be excluded.

Formally, the most powerful test of size alpha rejects H0 whenever the likelihood ratio exceeds a threshold c, and it accepts H0 whenever the likelihood ratio is below c. The threshold c is chosen so that the probability of rejection under H0 is exactly alpha, if possible. On the boundary where p1(x) equals c p0(x), the test may randomize: it rejects with some probability gamma that is selected to hit the size alpha exactly. This randomization is needed in discrete problems where the boundary has positive probability and no deterministic choice can achieve every possible value of alpha. In continuous problems the boundary often has probability zero, and a non-randomized threshold rule typically attains the desired size.

The proof of optimality is an elegant exchange argument. Take any other test whose size is at most alpha. Compare the difference in power between the likelihood-ratio test and this competitor. On the set where p1(x) is greater than c p0(x), the likelihood-ratio test rejects with probability one, so its rejection probability is at least as large as that of the competitor. On the set where p1(x) is less than c p0(x), the likelihood-ratio test rejects with probability zero, so its rejection probability is at most that of the competitor, but because the difference p1 minus c p0 is negative there, this contributes nonnegatively to the power difference. On the boundary the difference is zero. Adding the size constraint, which says the competitor does not use more null probability than the likelihood-ratio test, shows that the likelihood-ratio test has at least as much power as any other test of the same size.

This result is more than a recipe for constructing tests. It explains why many classical tests take the form they do. A z-test for a normal mean compares the sample mean to a threshold; that threshold is exactly the point where the likelihood ratio under the simple null and simple alternative equals a critical value. A t-test, when viewed as the uniformly most powerful unbiased test in a one-parameter exponential family, inherits its structure from the same likelihood-ratio ordering. Even when the alternative is composite, the Neyman-Pearson lemma motivates statistics such as the generalized likelihood ratio, which locally approximates the optimal simple-versus-simple comparison.

The lemma also clarifies the conceptual role of the p-value. A p-value is the smallest size alpha at which the observed data would lead to rejection. It is a transformation of the likelihood ratio that orders the sample space in the same way, and it is useful precisely because that ordering is the one that maximizes power against the specified alternative. Reporting a p-value without specifying the alternative can still be informative, but its optimality property comes from the Neyman-Pearson argument.

Another important consequence is the irrelevance of transformations. If I apply a one-to-one measurable transformation to the data, the likelihood ratio transforms in a way that preserves the ordering. The optimal test based on the transformed data is the same as the optimal test based on the original data. This invariance is not shared by ad hoc procedures that reject on, say, the absolute deviation from a sample median, unless that procedure happens to coincide with a likelihood-ratio test in the model at hand.

The Neyman-Pearson lemma also sets the boundary of what is possible. If I demand a size smaller than alpha, my maximum power cannot exceed the power of the size-alpha likelihood-ratio test. If I demand a power higher than that maximum, I must accept a larger size. This tradeoff curve, called the receiver operating characteristic or ROC curve, is traced out by varying the threshold c. At one extreme, when c is zero, I always reject and the test has size one and power one. At the other extreme, when c is infinity, I never reject and the test has size zero and power zero. The curve is concave, and its slope at any point is the threshold that generates that operating point.

In practice I may not know the exact alternative P1, or the null may not be simple. The lemma still serves as a benchmark. A test that is most powerful against a representative alternative, or that has high power against a broad class of alternatives, is often judged by how close it comes to the Neyman-Pearson envelope. In signal detection, for example, the matched filter is the likelihood-ratio detector for a known signal in additive white Gaussian noise, and its optimality follows immediately from the lemma.

I will now illustrate the lemma with a small simulation. I consider a test that observes a single real number X and must decide between H0: X is standard normal and H1: X is normal with mean mu and variance one. The likelihood ratio is proportional to exp(mu X - mu^2/2), which is a monotone function of X when mu is positive. Therefore the most powerful test rejects H0 when X exceeds a threshold. I simulate many draws under both hypotheses, implement the threshold test at a target size, and compare its empirical power to the power of a naive test that rejects when |X| exceeds a different threshold chosen to match the same size. The naive two-sided test is natural if I did not know the direction of the alternative, but against the one-sided alternative it wastes size on the left tail, where the likelihood ratio is small. The simulation shows that the likelihood-ratio test achieves higher empirical power while controlling the empirical size.

```python
import numpy as np

np.random.seed(0)

mu = 0.5
alpha = 0.05
n = 200_000

x0 = np.random.normal(loc=0.0, scale=1.0, size=n)
x1 = np.random.normal(loc=mu, scale=1.0, size=n)

# Most powerful test for H0: N(0,1) vs H1: N(mu,1) with mu > 0.
# Likelihood ratio is monotone in x, so reject when x > c.
c_np = np.quantile(x0, 1.0 - alpha)
reject_np_h0 = x0 > c_np
reject_np_h1 = x1 > c_np
size_np = reject_np_h0.mean()
power_np = reject_np_h1.mean()

# Naive two-sided test calibrated to the same empirical size.
c_two = np.quantile(np.abs(x0), 1.0 - alpha)
reject_two_h0 = np.abs(x0) > c_two
reject_two_h1 = np.abs(x1) > c_two
size_two = reject_two_h0.mean()
power_two = reject_two_h1.mean()

# Theoretical power of the Neyman-Pearson test.
from math import erfc
power_theory = 0.5 * erfc((c_np - mu) / np.sqrt(2))

print(f"Neyman-Pearson threshold: {c_np:.4f}")
print(f"NP empirical size: {size_np:.4f} (target {alpha})")
print(f"NP empirical power: {power_np:.4f}")
print(f"NP theoretical power: {power_theory:.4f}")
print(f"Two-sided threshold: {c_two:.4f}")
print(f"Two-sided empirical size: {size_two:.4f}")
print(f"Two-sided empirical power: {power_two:.4f}")

# Verify the likelihood-ratio ordering directly.
log_lr = mu * x1 - 0.5 * mu**2
print(f"Mean log-likelihood-ratio for rejected NP points: {log_lr[reject_np_h1].mean():.4f}")
print(f"Mean log-likelihood-ratio for accepted NP points: {log_lr[~reject_np_h1].mean():.4f}")
```

The Neyman-Pearson lemma remains the foundational statement of optimality in hypothesis testing. It turns the vague idea that a test should be powerful into a precise optimization problem, and it shows that the likelihood ratio is the natural quantity for ordering observations. Whenever I design or evaluate a statistical test, I am either using the lemma directly or comparing my procedure to the benchmark it provides.
