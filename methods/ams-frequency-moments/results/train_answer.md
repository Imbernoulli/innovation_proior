A stream a_1, ..., a_m over a universe [n] defines a frequency vector (m_1, ..., m_n), where m_i counts how often value i appears. The kth frequency moment is F_k = sum_i m_i^k. The moment we care about most is F_2, the sum of squared frequencies, because it measures skew and equals the size of a self-join on the stream key. The useful computational model is a single left-to-right pass with working memory far smaller than the full histogram.

The histogram baseline stores a counter for every possible value, costing Theta(n log m) bits. That linear dependence on the universe size n is the barrier: for large n the counters no longer fit in fast memory. Earlier randomized summaries show that randomness can replace exact state for counting stream length or distinct values, but those methods discard the identity of duplicates and therefore cannot capture squared frequencies, which depend on collisions between equal items rather than on totals.

A natural fallback is to sample a stream position uniformly. If the sampled position holds value i and r is the number of remaining occurrences of i from that position onward, then m(r^k - (r-1)^k) is an unbiased estimator of F_k, because the chosen occurrence is uniformly one of the m_i occurrences from the end and the terms telescope. Reservoir sampling removes the need to know m in advance. However, the variance of this position-sampling estimator is too high for F_2: it only reduces space to about n^{1/2}, not to the logarithmic space we want. A different structural idea is needed.

The AMS frequency-moment estimator, introduced by Alon, Matias, and Szegedy, solves the problem by maintaining a random linear projection of the frequency vector. For the general case it uses the position-sampling estimator inside a median-of-means procedure. For the second moment it does something far tighter. Choose a four-wise-independent random sign function epsilon from [n] to {-1, +1} and maintain the single scalar Z = sum_i epsilon_i m_i. As each stream item a_j arrives, update Z by adding epsilon(a_j). At the end output Z^2.

Expanding Z^2 gives sum_i m_i^2 plus sum_{i != j} epsilon_i epsilon_j m_i m_j. The first sum is exactly F_2. Pairwise independence with zero-mean signs makes every cross term vanish in expectation, so E[Z^2] = F_2. The variance is handled by a fourth-moment argument, which is why four-wise independence is exactly the right amount of randomness. With four-wise-independent signs, Var(Z^2) = 2(F_2^2 - F_4) <= 2 F_2^2, a constant relative variance.

To obtain a (lambda, delta)-relative-error estimate, average Theta(lambda^{-2}) independent copies of Z^2 and take the median of Theta(log(1/delta)) such averages. Chebyshev controls the variance of each average, and the median amplifies the success probability by a Chernoff bound. The total space is O(lambda^{-2} log(1/delta) (log n + log m)) bits, because each sketch stores one scalar and the sign family has a logarithmic seed.

The signs can be generated from a small seed using finite-field polynomial constructions or BCH orthogonal-array constructions; full independence over all n items is unnecessary. The key insight is to stop storing frequencies and instead store a random projection that can be maintained in one pass. Squaring the projection makes equal values reinforce on the diagonal while unrelated values cancel in expectation.

The finished estimator is exactly this. Fix a family of sign functions $\epsilon : [n] \to \{-1, +1\}$ that is four-wise independent, drawn from an $O(\log n)$-bit seed via a degree-three polynomial over a finite field (or an equivalent BCH/orthogonal-array construction). Maintain a single running scalar $Z$, initialized to zero; on each stream item $a_j$ update

$$Z \leftarrow Z + \epsilon(a_j),$$

and at the end of the pass output $X = Z^2$. Because $\epsilon_i^2 = 1$ and the cross terms cancel under pairwise independence, $\mathbb{E}[X] = F_2$, and the four-wise independence of the sign family gives

$$\operatorname{Var}(X) = 2\left(F_2^2 - F_4\right) \le 2F_2^2.$$

A single copy of $X$ only has constant relative variance, so it is not yet a $(\lambda,\delta)$-guarantee. To get one, run $s_1 = \Theta(\lambda^{-2})$ independent copies of the sketch in parallel, each with its own independently drawn sign family and its own running $Z_{r,c}$, and average the $s_1$ resulting values of $X$; Chebyshev's inequality then bounds that average's failure probability by a constant. Repeat this whole averaging block $s_2 = \Theta(\log(1/\delta))$ times independently and output the median of the $s_2$ averages; a Chernoff bound over the $s_2$ blocks drives the failure probability down to $\delta$. The complete estimator is

$$Y = \operatorname*{median}_{1 \le r \le s_2} \left( \frac{1}{s_1} \sum_{c=1}^{s_1} Z_{r,c}^2 \right), \qquad \Pr\big[\,|Y - F_2| > \lambda F_2\,\big] \le \delta,$$

maintained with total space

$$O\!\left(\lambda^{-2} \log(1/\delta)\,(\log n + \log m)\right)$$

bits: one scalar $Z_{r,c}$ per one of the $s_1 s_2$ parallel sketches, each scalar itself only $O(\log m)$ bits since it is a running sum bounded by the stream length, plus the $O(\log n)$-bit seed that generates each sketch's four-wise-independent signs. That is the whole discovery — replace the histogram with $s_1 s_2 = O(\lambda^{-2}\log(1/\delta))$ parallel copies of a single signed linear projection, square each one, and let median-of-means turn a constant-relative-variance building block into a genuine relative-error guarantee, at a total cost of $O(\log n + \log m)$ per copy instead of the $\Theta(n \log m)$ bits the histogram demanded.
