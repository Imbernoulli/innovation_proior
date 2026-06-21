## Dense Recurrence

The available dense recurrence theorem says that any positive-density subset of the integers contains arithmetic progressions of every fixed length. In weighted form, a bounded function `0 <= f <= 1` with positive mean has a positive normalized count of `k`-term progressions.

The primes have density up to `N` of about `1/log N`, so their upper density in the integers is zero. The quantitative bounds available for dense recurrence are stated for sets of positive density.

## Local Arithmetic Bias

The primes are distributed differently from a uniform random subset. Apart from finitely many exceptions, they avoid `0 mod p` for every small prime `p`, and more generally occupy only reduced residue classes modulo fixed moduli.

This local congruence structure is one of the standard features of the primes, captured by the singular-series corrections that appear in counts of prime patterns.

## Weighted Counting Language

The natural counting object is not just the set of primes but a weighted function. The von Mangoldt function assigns weight `log p` at prime powers and has average `1 + o(1)` by the prime number theorem.

Arithmetic progressions can be counted by averages such as `E_{x,r} f(x)f(x+r)...f(x+(k-1)r)` on a cyclic group. The dense theorem supplies a lower bound for this average when the weight is bounded by `1`.

## Sieve Weights

Sieve weights give nonnegative envelopes around primes and almost primes without requiring a full prime-tuples theorem. Truncated divisor sums are especially relevant because they approximate the von Mangoldt function while having correlations that can be estimated.

The setting joins three standing bodies of technique: bounded dense recurrence for progression counts, the local congruence structure of the primes, and sieve-built majorants for prime weights. The question is how progressions of length `k` are distributed among the primes themselves.
