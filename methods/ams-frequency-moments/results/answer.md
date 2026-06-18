# AMS Frequency-Moment Estimation

Let a stream over `[n]` define frequencies `m_i`. For `k > 0`, write `F_k = sum_i m_i^k`; `F_0` is the number of nonzero frequencies.

## General Position-Sampling Estimator

For `k >= 1`, choose a uniformly random stream position `p`. Let `a_p = l`, and let

`r = |{q >= p : a_q = l}|`.

Output

`X = m(r^k - (r - 1)^k)`.

Conditioned on choosing value `i`, the sampled occurrence is uniformly one of that value's `m_i` occurrences from the end, so

`sum_{c=1}^{m_i} (c^k - (c-1)^k) = m_i^k`.

Thus

`E[X] = sum_i (m_i / m) * m * (1 / m_i) * sum_{c=1}^{m_i}(c^k - (c-1)^k) = F_k`.

The second-moment calculation gives

`Var(X) <= E[X^2] <= k F_1 F_{2k-1} <= k n^{1-1/k} F_k^2`.

Reservoir sampling implements the uniform position choice when `m` is not known in advance. Median-of-means gives a one-pass `(lambda, delta)` estimator using

`O(k lambda^-2 log(1/delta) n^{1-1/k} (log n + log m))` bits.

## Special `F_2` Sketch

Choose a four-wise-independent random sign function `epsilon: [n] -> {-1, +1}`. Maintain one scalar

`Z = sum_i epsilon_i m_i`

by updating `Z <- Z + epsilon_{a_j}` for each stream item `a_j`. Output `X = Z^2`.

Unbiasedness:

`Z^2 = sum_i m_i^2 + sum_{i != j} epsilon_i epsilon_j m_i m_j`,

and pairwise independence with zero-mean signs kills the cross terms, so `E[Z^2] = F_2`.

Variance:

Four-wise independence is what controls `E[Z^4]`. Terms with any sign appearing an odd number of times vanish. The surviving terms are:

`E[Z^4] = F_4 + 6 sum_{i<j} m_i^2 m_j^2`.

With `S = sum_{i<j} m_i^2 m_j^2`, `F_2^2 = F_4 + 2S`, hence

`Var(Z^2) = 4S = 2(F_2^2 - F_4) <= 2F_2^2`.

Average `Theta(lambda^-2)` independent copies to get constant success by Chebyshev, then take the median of `Theta(log(1/delta))` such averages. Total space:

`O(lambda^-2 log(1/delta) (log n + log m))` bits.

The signs do not require full randomness: four-wise-independent families from BCH/orthogonal-array constructions, or equivalent finite-field polynomial constructions, have `O(log n)`-bit descriptions.

## Scientific Move

The breakthrough is the random linear view of a frequency vector. The algorithm never stores `m_i`; it stores a small random projection that can be updated in one pass. Squaring the projection makes equal-item contributions reinforce on the diagonal, while unrelated item interactions cancel in expectation.

This is more than applying Chernoff bounds to sampled counts. Chernoff only amplifies the final median-of-means. The decisive work is designing the estimator and proving the fourth-moment variance bound with exactly four-wise independence.

No stable-law coefficients are part of this construction: the `F_2` estimator uses Rademacher signs and a fourth-moment argument.
