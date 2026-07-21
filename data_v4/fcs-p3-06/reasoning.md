Two numbers in the constraints decide this whole problem before I pick any algorithm: `n_i` goes up to `10^7`, and the prime `p` goes up to `2^31 - 1`. Everything else — up to `10^5` queries, a single shared `p`, the definition `D(0) = 1`, `D(1) = 0` — sits underneath those two.

The prime is the first thing I have to respect in code. `p` itself fits in 32 bits, but a *product* of two residues, each up to `p - 1 ~ 2.1 * 10^9`, reaches about `4.6 * 10^18` — well past the 32-bit range, though comfortably inside signed 64-bit's `~9.2 * 10^18` ceiling. So every residue and every intermediate product lives in `long long`, and I reduce after each multiply. An `int` anywhere on this path is a silent wrong answer on any test with a large prime.

The `10^7` is what makes the tempting shortcut a wrong answer. The small counts `D(0..7) = 1, 0, 1, 2, 9, 44, 265, 1854` are tidy and the sample stops at `n <= 7`, which invites hardcoding a prefix `D[0..K]` and indexing into it. But there is no storable `K` that covers `[0, 10^7]` as literal constants, and the moment I precompute those entries I am running the algorithm rather than storing it. A single hidden query `n = 10^7` has no slot in any table I could ship. So I need the general recurrence, sized to whatever the queries demand.

Count derangements of `{1, ..., n}` by where element `n` is sent. It cannot go home, so it goes to one of `n - 1` positions `j`. Split on element `j`: either `j` lands on position `n`, so `n` and `j` have swapped and the remaining `n - 2` elements form a derangement among themselves (`D(n - 2)`); or `j` avoids position `n`, in which case the `n - 1` elements other than `n`, each forbidden from its home slot (with position `n` playing the forbidden slot for `j`), form a derangement of `n - 1` items (`D(n - 1)`). The `(n - 1)` choice of `j` multiplies both disjoint cases:

`D(n) = (n - 1) * (D(n - 1) + D(n - 2))`, for `n >= 2`, with `D(0) = 1`, `D(1) = 0`.

This is `O(1)` per step and, unlike the inclusion-exclusion form `D(n) = n! * sum_k (-1)^k / k!`, uses only additions and multiplications — no modular inverse to get right under the prime. The one place it can silently go wrong is the seeds: with `D(0) = 1`, `D(1) = 0` the recurrence gives `D(2) = 1`, `D(3) = 2`, then `9, 44, 265, 1854`, matching the sample; seeding `D(0) = 0` instead would shift the entire sequence.

For the batch: the `10^5` queries share one `p`, so recomputing per query would be `O(T * maxN) = 10^12`, hopeless. Instead read every `n_i`, take `maxN = max(n_i)`, run the recurrence once up to `maxN` into `der[k] = D(k) mod p`, then answer each query by an `O(1)` lookup. That is `O(maxN + T)` time; the array is `maxN + 1` `long long`s, about `80 MB` at `maxN = 10^7`, under the `256 MB` limit.

The one pitfall the constraints genuinely invite is the degenerate batch where *every* query is `n = 0`. Then `maxN = 0` and `der` has size `1`, but the base-case setup wants to write both `der[0]` and `der[1]` — and `der[1]` is out of bounds. An all-`n=0` batch run against the oracle is what surfaces it. The fix is to guard each base-case write by `maxN`: write `der[1]` only when `maxN >= 1`. I also do the reduction in two steps so the product provably stays in range:

```
long long coeff = (n - 1) % p;
long long inner = (der[n - 1] + der[n - 2]) % p;
der[n] = (coeff * inner) % p;
```

Here `coeff < p` and `inner < p`, so `coeff * inner < p^2 <= (2^31 - 1)^2 ~ 4.6 * 10^18`, inside `long long`; and `der[n-1] + der[n-2] < 2p <= 2^32` fits before the reduction.

For an independent check I differential-test against an oracle grounded in the *definition* rather than this recurrence, so agreement is not circular: inclusion-exclusion `D(n) = sum_k (-1)^k C(n,k) (n-k)!` in exact integers, cross-checked for small `n` by literally enumerating permutations and counting fixed-point-free ones. Over hundreds of random cases spanning small primes `2, 3, 5`, large primes `10^9 + 7`, `998244353`, `2^31 - 1`, and `n` from tiny to a few thousand — plus the edges `n = 0`, `n = 1`, and the all-`n=0` batch that exposed the out-of-bounds write — there are zero mismatches. The `10^7` case, which no table could ever reach, runs the single sweep in about `0.12 s` at `~81 MB`, within both limits.

Last detail on output: `10^5` lines through the stream one at a time pays per-line overhead, so I build the whole output into one `string` and flush it once. The full module is in the answer.
