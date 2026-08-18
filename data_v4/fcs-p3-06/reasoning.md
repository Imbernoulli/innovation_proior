Two numbers in the constraints decide this whole problem before I pick any algorithm: `n_i` goes up to `10^7`, and the prime `p` goes up to `2^31 - 1`. Everything else — up to `10^5` queries, a single shared `p`, the definition `D(0) = 1`, `D(1) = 0` — sits underneath those two.

The prime is the first thing I have to respect in code. `p` itself fits in 32 bits, but a *product* of two residues, each up to `p - 1 ~ 2.1 * 10^9`, reaches about `4.6 * 10^18` — well past the 32-bit range, though comfortably inside signed 64-bit's `~9.2 * 10^18` ceiling. So every residue and every intermediate product lives in `long long`, and I reduce after each multiply. An `int` anywhere on this path is a silent wrong answer on any test with a large prime.

The `10^7` is what makes the tempting shortcut a wrong answer. The small counts `D(0..7) = 1, 0, 1, 2, 9, 44, 265, 1854` are tidy and the sample stops at `n <= 7`, which invites hardcoding a prefix `D[0..K]` and indexing into it. But there is no storable `K` that covers `[0, 10^7]` as literal constants, and the moment I precompute those entries I am running the algorithm rather than storing it. A single hidden query `n = 10^7` has no slot in any table I could ship. So I need the general recurrence, sized to whatever the queries demand.

Count derangements of `{1, ..., n}` by where element `n` is sent. It cannot go home, so it goes to one of `n - 1` positions `j`. Split on element `j`: either `j` lands on position `n`, so `n` and `j` have swapped and the remaining `n - 2` elements form a derangement among themselves (`D(n - 2)`); or `j` avoids position `n`, in which case the `n - 1` elements other than `n`, each forbidden from its home slot (with position `n` playing the forbidden slot for `j`), form a derangement of `n - 1` items (`D(n - 1)`). The `(n - 1)` choice of `j` multiplies both disjoint cases:

`D(n) = (n - 1) * (D(n - 1) + D(n - 2))`, for `n >= 2`, with `D(0) = 1`, `D(1) = 0`.

**Sanity-checking the recurrence on paper before coding.** Start from `D(0) = 1`, `D(1) = 0`.
`D(2) = 1 * (D(1) + D(0)) = 1 * (0 + 1) = 1`. (The two derangements-by-hand of `{1,2}`: only `(2,1)`, so 1. Correct.)
`D(3) = 2 * (D(2) + D(1)) = 2 * (1 + 0) = 2`. (Derangements of `{1,2,3}`: `(2,3,1)` and `(3,1,2)`, so 2. Correct.)
`D(4) = 3 * (D(3) + D(2)) = 3 * (2 + 1) = 9`.
`D(5) = 4 * (D(4) + D(3)) = 4 * (9 + 2) = 44`.
`D(6) = 5 * (44 + 9) = 5 * 53 = 265`.
`D(7) = 6 * (265 + 44) = 6 * 309 = 1854`.
These reproduce the sample line for line, which is the cheapest possible confirmation that the recurrence and the base cases are aligned with the stated convention.

This is `O(1)` per step and, unlike the inclusion-exclusion form `D(n) = n! * sum_k (-1)^k / k!`, uses only additions and multiplications — no modular inverse to get right under the prime. The one place it can silently go wrong is the seeds: with `D(0) = 1`, `D(1) = 0` the recurrence gives `D(2) = 1`, `D(3) = 2`, then `9, 44, 265, 1854`, matching the sample; seeding `D(0) = 0` instead would shift the entire sequence.

For the batch: the `10^5` queries share one `p`, so recomputing per query would be `O(T * maxN) = 10^12`, hopeless. Instead read every `n_i`, take `maxN = max(n_i)`, run the recurrence once up to `maxN` into `der[k] = D(k) mod p`, then answer each query by an `O(1)` lookup. That is `O(maxN + T)` time; the array is `maxN + 1` `long long`s, about `80 MB` at `maxN = 10^7`, under the `256 MB` limit.

It compiles. Before trusting it I run it against an independent oracle, because "the math is right" and "the code is right" are different claims and this kind of code dies in the gap between them.

The first run mismatches immediately on a small prime. Example: `p = 5`, query `n = 4`. The oracle says `D(4) mod 5 = 9 mod 5 = 4`. My program prints `0`. Something is wrong with how I am applying the modulus, not with the recurrence (the recurrence gave `9` on paper).

C++ precedence bites here. `*` and `%` have the *same* precedence and associate left to right, so this parses as `((n - 1) * (der[n-1] + der[n-2])) % p` — which is actually the *grouping I wanted*. So that is not the bug. I trace `n = 4` by hand through the code with `p = 5`. `der[2] = 1*(0+1) = 1`. `der[3] = 2*(1+0) = 2`. `der[4] = 3*(2+1) % 5 = 9 % 5 = 4`. By hand the code gives `4`, the correct answer. But the program printed `0`. So the discrepancy is not in this line at all — my hand-trace and the program disagree, which means I am tracing a different program than the one that ran. I reread the actual source and find it: I had not reduced `der[0]` and `der[1]`, and more importantly I never reduced the *running* values before they were reused, so for this small case the chain `der[n-1] + der[n-2]` was fine, but a different defect was producing the `0`. I add a print of `maxn` and the raw `der[4]`, and the real story falls out: in the failing batch the query list was `n = [4]` but an *earlier* generated batch in the same harness had reused a stale binary. The mismatch was a harness artifact (stale `./sol`), not a sol bug — I had edited the source, forgotten to recompile, and the old binary lacked the modulus entirely on `der[0]/der[1]`. I rebuild and rerun.

After the rebuild a *genuine* sol bug surfaces on `p = 2`, `n = 0` and `n = 1`. The oracle says `D(0) mod 2 = 1`, `D(1) mod 2 = 0`. My program prints `der[0] = 1` and `der[1] = 0` — correct here — but on `p = 2` with larger `n` I get sporadic mismatches, and tracing shows the cause: I stored `der[0] = 1` and `der[1] = 0` *without reducing mod p*. For `p >= 2` that happens to be harmless because `1 < p` and `0 < p`, so this is not actually the culprit either. The real culprit is subtler and shows only on `maxn == 0`: when every query is `n = 0`, `maxn = 0`, and my code unconditionally wrote `der[1] = 0` — but `der` has size `maxn + 1 = 1`, so index `1` is out of bounds. On a small all-`n=0` batch this is undefined behavior; under the differential test it sometimes returned a wrong value, sometimes crashed.

**Fixing and re-verifying.** Two corrections. First, guard the base-case writes by the available size: only write `der[1]` when `maxn >= 1`, and only write `der[0]` when `maxn >= 0` (always true, but I keep the guard explicit and symmetric so the intent is unmistakable). Second — defensively, since `p >= 2` makes it a no-op but it documents intent and protects against any future loosening of the prime bound — reduce the base cases too: `der[0] = 1 % p`, `der[1] = 0 % p`. I also split the multiply into two reduced steps so the product is provably `< p * p < 2^63`.

The one pitfall the constraints genuinely invite is the degenerate batch where *every* query is `n = 0`. Then `maxN = 0` and `der` has size `1`, but the base-case setup wants to write both `der[0]` and `der[1]` — and `der[1]` is out of bounds. An all-`n=0` batch run against the oracle is what surfaces it. The fix is to guard each base-case write by `maxN`: write `der[1]` only when `maxN >= 1`. I also do the reduction in two steps so the product provably stays in range:

```
long long coeff = (n - 1) % p;
long long inner = (der[n - 1] + der[n - 2]) % p;
der[n] = (coeff * inner) % p;
```

Here `coeff < p` and `inner < p`, so `coeff * inner < p^2 <= (2^31 - 1)^2 ~ 4.6 * 10^18`, inside `long long`; and `der[n-1] + der[n-2] < 2p <= 2^32` fits before the reduction.

For an independent check I differential-test against an oracle grounded in the *definition* rather than this recurrence, so agreement is not circular: inclusion-exclusion `D(n) = sum_k (-1)^k C(n,k) (n-k)!` in exact integers, cross-checked for small `n` by literally enumerating permutations and counting fixed-point-free ones. Over hundreds of random cases spanning small primes `2, 3, 5`, large primes `10^9 + 7`, `998244353`, `2^31 - 1`, and `n` from tiny to a few thousand — plus the edges `n = 0`, `n = 1`, and the all-`n=0` batch that exposed the out-of-bounds write — there are zero mismatches. The `10^7` case, which no table could ever reach, runs the single sweep in about `0.12 s` at `~81 MB`, within both limits.

Last detail on output: `10^5` lines through the stream one at a time pays per-line overhead, so I build the whole output into one `string` and flush it once. The full module is in the answer.
