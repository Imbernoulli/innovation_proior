The two range lines are the whole problem. `N` runs to `10^18`, so I cannot walk the string length
even once per query, let alone across `10^5` queries — I need an evaluation with a `log N` factor.
And `p` runs to `10^18`, so the product of two residues, each near `10^18`, reaches `10^36` and
overflows a 64-bit integer by eighteen orders of magnitude; every modular multiply has to pass through
a wider intermediate or the large-modulus tests silently return garbage. Both choices — `unsigned long
long` for the state, `__uint128_t` only inside the multiply — are forced before I even know the
recurrence, so I fix them now.

Let `f(N)` be the count. Enumerating the smallest lengths by hand anchors everything else: `f(0) = 1`
(only the empty string, vacuously valid), `f(1) = 2` (`0`, `1`), `f(2) = 3` (`00, 01, 10`; `11`
excluded), `f(3) = 5` (`000, 001, 010, 100, 101` valid, `011, 110, 111` not). Length 4 gives 8, length
5 gives 13 — the sequence is `1, 2, 3, 5, 8, 13, 21, ...`, consecutive Fibonacci numbers, with
`f(N) = Fib(N+2)`.

That tidiness is the trap. The cheap move is a constant table
`table[N] % p`, which reproduces every sample line and cannot survive the scored range. `Fib`
grows like `phi^N`, so the raw value `f(90)` already exceeds `10^18`, and `f(10^18)` has on the order of
`2 * 10^17` decimal digits; there is no `table[10^18]` to store, and no way to hold the integer even if
there were. The hidden tests are described as clustering near `N = 10^18`, exactly where any feasible
table indexes out of bounds or has no entry. So the samples are bait: I derive the general recurrence
and evaluate it in `O(log N)`, with no table.

To get a relation valid for all `N`, take a valid string of length `N >= 2` and split on its last
character. If it ends in `0`, dropping that character leaves any valid string of length `N - 1`,
unconstrained — `f(N - 1)` of them. If it ends in `1`, the character before must be `0` (else `11`), so
the string ends `01`; dropping both leaves any valid string of length `N - 2`, and prepending `01`
preserves validity — `f(N - 2)` of them. The two cases are disjoint and exhaustive, so
`f(N) = f(N - 1) + f(N - 2)` for `N >= 2`, with bases `f(0) = 1`, `f(1) = 2`. It reproduces `f(2) = 3`,
`f(3) = 5`, `f(4) = 8`, `f(5) = 13` — the values I enumerated, so I trust it.

A constant-coefficient linear recurrence is a matrix power, which is what buys the `log N`. With state
`v_k = [f(k), f(k-1)]^T` and `M = [[1, 1], [1, 0]]`, we have `v_{k+1} = M v_k`, so `v_n = M^{n-1} v_1`
with `v_1 = [f(1), f(0)]^T = [2, 1]^T`, and
`f(n) = (M^{n-1})[0][0] * f(1) + (M^{n-1})[0][1] * f(0)`. Binary exponentiation raises `M` in
`O(log N)` — about 60 squarings at `N = 10^18`, a handful of modular multiplies each, comfortably inside
two seconds for `10^5` queries.

The exponent carries an off-by-one that the indexing invites. Because the state starts
at `v_1`, reaching `v_n` takes `M^{n-1}`, not `M^n` — but "raise `M` to the length" reads as `M^n`, and
`M^n` computes `v_{n+1}`. At `n = 2`, `M^n = M^2 = [[2,1],[1,1]]` would give `2*2 + 1*1 = 5`, which is
`f(3)`, not `f(2) = 3`; the shift is exactly one index. With `n - 1` it gives `M^1: 1*2 + 1*1 = 3`,
correct, and `n = 5` gives `M^4 = [[5,3],[3,2]]: 5*2 + 3*1 = 13`, correct. So the matrix path uses
exponent `n - 1`.

That same exponent settles the base cases. `n = 0` would need `M^{-1}`, which the exponentiation cannot
produce, so `n = 0` must be handled directly (return `1 % p`). `n = 1` gives exponent `0`, i.e. the
identity, which does return `f(1) = 2` correctly — but I peel it off as well, so the matrix path only
runs for `n >= 2`, where `n - 1 >= 1` is unambiguous.

For the multiply: with `p` up to `10^18` a residue is near `2^60` and the product of two is near
`2^120`, so `mulmod` casts to `__uint128_t`, multiplies, and reduces. Each 2x2 entry is a sum of two
such products; subtracting `MOD` after each add keeps the accumulator below `2 * MOD < 2^64`. Reducing
`f0 = 1 % p` and `f1 = 2 % p` up front quietly handles `p = 1`: there every residue and every matrix
entry is `0`, so every answer is `0` — correct, with no special branch.

Hand-traces alone will not settle the near-`10^18` cases, so I cross-check my program against oracles
that share no code path with it: an exhaustive `2^N`-bitmask enumerator counting masks with
`mask & (mask >> 1) == 0` for small `N`; an exact arbitrary-precision DP (`a, b = b, a + b`, reduced only
at the end) for small and medium `N`; and, for the `N` the brute force cannot reach, an independent
fast-doubling Fibonacci reference so the near-`10^18` edge cases are still checked. Across thousands of
mixed cases — spanning `p = 1`, tiny primes, `10^9`-scale primes, and full-range `10^18` moduli, and
including `N = 10^18`, `N = 10^18 - 1`, and `N = 2^60` — every output matched, including in the large-`N`
regime where the tempting table would have returned nothing.

So the program is matrix exponentiation of `f(N) = f(N-1) + f(N-2)` with `__uint128_t` modular
multiplication and `n = 0, 1` peeled off directly — not the Fibonacci table the samples invited.
