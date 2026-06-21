# Neyman-Pearson Lemma

Let `H0:P0` and `H1:P1` be simple hypotheses with densities `p0` and `p1` with respect to a common measure. A randomized test is a measurable function `phi(x)` with values in `[0,1]`, where `phi(x)` is the probability of rejecting `H0` after observing `x`. Its size is `E0 phi = integral phi p0 dmu`, and its power against `H1` is `E1 phi = integral phi p1 dmu`.

For a prescribed size `alpha`, the most powerful level-`alpha` test rejects on the largest values of the likelihood ratio `p1(x)/p0(x)`, treating `p0=0<p1` as infinite ratio and ignoring points where both densities are zero. Equivalently, avoid division by `p0`: there is a constant `c >= 0` and boundary randomization such that

```text
phi*(x) = 1  if p1(x) > c p0(x),
phi*(x) = 0  if p1(x) < c p0(x),
0 <= phi*(x) <= 1 on {x : p1(x) = c p0(x)},
```

with boundary values chosen so `E0 phi* = alpha` for randomized tests. Neyman and Pearson's 1933 notation uses the reciprocal inequality `p0 <= k p1`; for `c > 0` this is the same critical region with `k = 1/c`.

For any competing test `psi` with `E0 psi <= E0 phi*`, compare powers:

```text
E1(phi* - psi)
= integral (phi* - psi)p1 dmu
= integral (phi* - psi)(p1 - c p0) dmu + c integral (phi* - psi)p0 dmu.
```

The first integral is nonnegative because `phi*` equals `1` where `p1 - c p0` is positive, equals `0` where it is negative, and the boundary contributes zero. The second integral is nonnegative because `c >= 0` and `E0 phi* >= E0 psi`. Therefore `E1 phi* >= E1 psi`.

The theorem's distinctive content is the ordering principle. Size is a budget measured under `P0`; power is the reward measured under `P1`. Moving a small amount of `P0` mass from a low `p1/p0` point to a high `p1/p0` point preserves size and increases power. A most powerful test is exactly a rejection rule with no such improvable inversion.
