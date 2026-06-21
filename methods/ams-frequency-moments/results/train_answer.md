A stream a_1, ..., a_m over a universe [n] defines a frequency vector (m_1, ..., m_n), where m_i counts how often value i appears. The kth frequency moment is F_k = sum_i m_i^k. The moment we care about most is F_2, the sum of squared frequencies, because it measures skew and equals the size of a self-join on the stream key. The useful computational model is a single left-to-right pass with working memory far smaller than the full histogram.

The histogram baseline stores a counter for every possible value, costing Theta(n log m) bits. That linear dependence on the universe size n is the barrier: for large n the counters no longer fit in fast memory. Earlier randomized summaries show that randomness can replace exact state for counting stream length or distinct values, but those methods discard the identity of duplicates and therefore cannot capture squared frequencies, which depend on collisions between equal items rather than on totals.

A natural fallback is to sample a stream position uniformly. If the sampled position holds value i and r is the number of remaining occurrences of i from that position onward, then m(r^k - (r-1)^k) is an unbiased estimator of F_k, because the chosen occurrence is uniformly one of the m_i occurrences from the end and the terms telescope. Reservoir sampling removes the need to know m in advance. However, the variance of this position-sampling estimator is too high for F_2: it only reduces space to about n^{1/2}, not to the logarithmic space we want. A different structural idea is needed.

The AMS frequency-moment estimator, introduced by Alon, Matias, and Szegedy, solves the problem by maintaining a random linear projection of the frequency vector. For the general case it uses the position-sampling estimator inside a median-of-means procedure. For the second moment it does something far tighter. Choose a four-wise-independent random sign function epsilon from [n] to {-1, +1} and maintain the single scalar Z = sum_i epsilon_i m_i. As each stream item a_j arrives, update Z by adding epsilon(a_j). At the end output Z^2.

Expanding Z^2 gives sum_i m_i^2 plus sum_{i != j} epsilon_i epsilon_j m_i m_j. The first sum is exactly F_2. Pairwise independence with zero-mean signs makes every cross term vanish in expectation, so E[Z^2] = F_2. The variance is handled by a fourth-moment argument, which is why four-wise independence is exactly the right amount of randomness. With four-wise-independent signs, Var(Z^2) = 2(F_2^2 - F_4) <= 2 F_2^2, a constant relative variance.

To obtain a (lambda, delta)-relative-error estimate, average Theta(lambda^{-2}) independent copies of Z^2 and take the median of Theta(log(1/delta)) such averages. Chebyshev controls the variance of each average, and the median amplifies the success probability by a Chernoff bound. The total space is O(lambda^{-2} log(1/delta) (log n + log m)) bits, because each sketch stores one scalar and the sign family has a logarithmic seed.

The signs can be generated from a small seed using finite-field polynomial constructions or BCH orthogonal-array constructions; full independence over all n items is unnecessary. The key insight is to stop storing frequencies and instead store a random projection that can be maintained in one pass. Squaring the projection makes equal values reinforce on the diagonal while unrelated values cancel in expectation.

```python
import math
import random
import statistics


class AMSF2Sketch:
    """One-pass (lambda, delta)-relative-error estimator for F_2.

    Uses a degree-3 polynomial over a prime field to produce four-wise
    independent signs; each sketch maintains the scalar Z = sum_i eps_i m_i.
    """

    def __init__(self, n, lambda_param=0.1, delta=0.05):
        # A Mersenne prime larger than any realistic universe size.
        self.p = (1 << 61) - 1
        self.group_size = max(1, int(math.ceil(4.0 / (lambda_param ** 2))))
        self.num_groups = max(1, int(math.ceil(math.log(1.0 / delta))))
        self.total = self.group_size * self.num_groups
        # Each sketch gets four random field coefficients.
        self.coeffs = [
            tuple(random.randrange(self.p) for _ in range(4))
            for _ in range(self.total)
        ]
        self.Z = [0] * self.total

    def _sign(self, x, coeff):
        a, b, c, d = coeff
        x_mod = x % self.p
        x2 = (x_mod * x_mod) % self.p
        x3 = (x2 * x_mod) % self.p
        v = (a * x3 + b * x2 + c * x_mod + d) % self.p
        return 1 if v < self.p // 2 else -1

    def update(self, item):
        for i, coeff in enumerate(self.coeffs):
            self.Z[i] += self._sign(item, coeff)

    def estimate(self):
        group_avgs = []
        for g in range(self.num_groups):
            start = g * self.group_size
            total = 0.0
            for i in range(start, start + self.group_size):
                total += self.Z[i] * self.Z[i]
            group_avgs.append(total / self.group_size)
        return statistics.median(group_avgs)


def estimate_f2(stream, n, lambda_param=0.1, delta=0.05):
    sketch = AMSF2Sketch(n, lambda_param=lambda_param, delta=delta)
    for item in stream:
        sketch.update(item)
    return sketch.estimate()


if __name__ == "__main__":
    # Small example: value 1 appears 10 times, value 2 appears 3 times.
    stream = [1] * 10 + [2] * 3
    n = 10
    true_f2 = 10 ** 2 + 3 ** 2
    approx = estimate_f2(stream, n, lambda_param=0.2, delta=0.05)
    print(f"true F_2 = {true_f2}, estimate = {approx:.2f}")
```
