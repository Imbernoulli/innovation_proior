## Zero-Density Obstacle

The available dense recurrence theorem says that any positive-density subset of the integers contains arithmetic progressions of every fixed length. In weighted form, a bounded function `0 <= f <= 1` with positive mean has a positive normalized count of `k`-term progressions.

The primes are outside that theorem's direct range. Their density up to `N` is about `1/log N`, so their upper density in the integers is zero. Existing quantitative bounds for dense recurrence are far too weak to compensate for that sparsity.

## Local Arithmetic Bias

The primes are not sparse in the same way a uniform random subset is sparse. Apart from finitely many exceptions, they avoid `0 mod p` for every small prime `p`, and more generally occupy only reduced residue classes modulo fixed moduli.

Any ambient object that looks uniform across residue classes cannot directly dominate the unmodified prime-counting weight by a fixed constant. The local congruence structure has to be normalized before a randomness statement can even be plausible.

## Weighted Counting Language

The natural counting object is not just the set of primes but a weighted function. The von Mangoldt function assigns weight `log p` at prime powers and has average `1 + o(1)` by the prime number theorem.

Arithmetic progressions can be counted by averages such as `E_{x,r} f(x)f(x+r)...f(x+(k-1)r)` on a cyclic group. The dense theorem supplies a lower bound for this average when the weight is bounded by `1`; the obstacle is that prime weights are sparse and unbounded.

## Sparse Ambient Candidate

A workable sparse ambient object would need two properties at once. It must be large enough to contain a positive relative density of the normalized prime weight, and it must be random enough that dense progression-counting arguments still behave correctly inside it.

That randomness cannot mean arbitrary randomness. It only has to control the finite systems of linear forms and repeated-shift correlations that arise when counting `k`-term progressions and when applying Cauchy-Schwarz repeatedly.

## Sieve Signal

Sieve weights give nonnegative envelopes around primes and almost primes without requiring a full prime-tuples theorem. Truncated divisor sums are especially relevant because they approximate the von Mangoldt function while having correlations that can be estimated.

The pre-method problem is therefore sharply posed: remove the small-modulus obstructions, find a sieve-built sparse envelope with the right finite pseudorandomness properties, and create a bridge from bounded dense recurrence to positive relative density inside that envelope.
