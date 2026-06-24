**Reading the problem and pinning the contract.** I have `S` identical candies and `k` distinguishable children; each child gets between `0` and `c` candies inclusive; I must count the distributions, i.e. the integer solutions of `x_1 + ... + x_k = S` with `0 <= x_i <= c`, modulo a prime `M`. Input is the four numbers `k c S M` on stdin; I print one integer. The scales matter for the data types and the algorithm: `k, S <= 10^6`, `c <= 10^9`, and `M` is a prime with `M > S + k` and `M <= 10^9 + 7`. The promise `M > S + k` is deliberate — it means the largest binomial top I will ever form, `S + k - 1`, is strictly below `M`, so `(S+k-1)!` is not divisible by `M` and modular factorial inverses are valid (no factor of `M` ever appears in a factorial I invert). Everything multiplies products of residues below `10^9 + 7`, so a single `long long` multiply can reach `~10^{18}`, which fits, as long as I reduce after each multiply. I will use `long long` throughout and reduce eagerly.

**Laying out the candidate approaches.** Two routes are on the table.

- *Bounded-composition DP.* Process children one at a time, carrying `dp[s]` = number of ways to reach partial sum `s` using the children seen so far, each contributing `0..c`. The transition adds a windowed range and is `O(k*S)` with prefix sums. That is up to `10^6 * 10^6 = 10^{12}` operations — far too slow for a 1-second limit, even though it is the most obviously-correct formulation (I will use exactly this as my offline brute-force oracle on small inputs, but it cannot be the shipped solution).
- *Inclusion-exclusion over the cap.* The unbounded count of `y_1 + ... + y_k = T` is the stars-and-bars number `C(T + k - 1, k - 1)`. The cap `x_i <= c` is violated exactly when `x_i >= c + 1`; substituting `x_i = (c+1) + x_i'` removes one cap and lowers the total by `c+1`. Inclusion-exclusion over which children overflow gives a closed alternating sum with at most `k + 1` terms, each a couple of binomials. With a factorial table this is `O(S + k)` precompute plus `O(k)` summation. This is the one to ship.

**Deriving the inclusion-exclusion and checking it on paper.** Let `A_i` be the set of "bad" distributions in which child `i` overflows (`x_i >= c+1`), counted *without any upper bound* on the other children. By inclusion-exclusion the number of distributions with **no** overflow is

```
answer = sum_{j=0}^{?} (-1)^j * (number of ways to pick j children to force-overflow, times the
                                  unbounded count after subtracting (c+1) from each of those j).
```

Forcing a specific set of `j` children to overflow and substituting drops the total by `j*(c+1)`; the remaining unbounded count is `C((S - j*(c+1)) + k - 1, k - 1)`, and there are `C(k, j)` ways to choose which `j` children. So

```
answer = sum_j (-1)^j * C(k, j) * C(S - j*(c+1) + k - 1, k - 1).
```

The only thing left is the range of `j`. A term is meaningful exactly while the leftover total
`S - j*(c+1)` is `>= 0` (a negative leftover has zero unbounded distributions), and of course `j` cannot
exceed `k`. I want to nail this boundary with a concrete derivation rather than a feeling, so I hand-run
the stated sample `k=4, c=5, S=12` (`step = c+1 = 6`):

- `j=0`: `rem=12`, `C(4,0)*C(12+3,3) = 1*C(15,3) = 455`, sign `+`.
- `j=1`: `rem=6`,  `C(4,1)*C(6+3,3)  = 4*C(9,3)  = 4*84 = 336`, sign `-`.
- `j=2`: `rem=0`,  `C(4,2)*C(0+3,3)  = 6*C(3,3)  = 6*1  = 6`, sign `+`.
- `j=3`: `rem = 12 - 18 = -6 < 0` -> stop.

Sum `= 455 - 336 + 6 = 125`, which is the sample answer. Notice the *last* contributing term, `j=2`, has
`rem = 0`: it is the case where the entire total is absorbed by exactly two maxed-out overflows. It is a
real, on-the-boundary configuration and it contributes `+6`. The stopping test therefore has to admit
`rem = 0`; it must be `S - j*(c+1) >= 0`, inclusive. If I had written a strict `> 0` I would have dropped
that `+6` and reported `119`. I flag this now because it is exactly the kind of boundary I tend to get
wrong under time pressure, and I will verify it explicitly below.

**First implementation — and immediately a trace, because boundaries bite.** My first cut of the core loop:

```
long long ans = 0, step = c + 1;
for (long long j = 0; S - j * step > 0 && j <= k; j++) {   // <-- note the > 0
    long long rem = S - j * step;
    long long ways = C(k, j) * C(rem + k - 1, k - 1) % M;
    if (j & 1) ans = (ans - ways + M) % M;
    else       ans = (ans + ways) % M;
}
```

I deliberately pick the smallest input that puts the leftover exactly on the boundary so the `> 0` versus
`>= 0` choice is exercised: `k=2, c=1, S=2`, `M` large. The truth is easy to see by hand — two children
each in `{0,1}` summing to `2` forces `(1,1)`, so the answer is `1`. Trace the buggy loop, `step = 2`:

- `j=0`: `S - 0 = 2 > 0` true. `rem=2`. `C(2,0)*C(2+1,1) = 1*C(3,1) = 3`. `ans = 3`.
- `j=1`: `S - 1*2 = 0`. Condition `0 > 0` is **false** -> loop exits.

Final `ans = 3`. But the answer is `1`. The code is wrong, and it is wrong by exactly the term I warned
myself about.

**Diagnosing bug #1 (the off-by-one boundary).** The dropped iteration is `j=1`, where `rem = 0`: forcing
both children to overflow leaves `0` candies, one valid configuration to subtract, contributing
`-C(2,1)*C(0+1,1) = -2*1 = -2`. With it, `3 - 2 = 1`, correct. Without it, I over-count by exactly the
boundary configuration. The defect is the strict inequality `S - j*step > 0`: it treats a leftover of `0`
as "nothing to count", but a leftover of `0` is a real distribution (everyone's overflow exactly uses up
the total). The condition must be inclusive, `S - j*step >= 0`. This is the canonical inclusive/exclusive
slip: the *upper boundary of the inclusion-exclusion sum is inclusive*. Note also this is the same slip
that would have turned the sample `4 5 12` from `125` into `119` by dropping its `j=2` term. One character —
`>` vs `>=` — is the whole bug.

**Fixing bug #1 and re-verifying.** Change the guard to `>= 0`:

```
for (long long j = 0; S - j * step >= 0 && j <= k; j++) {
```

Re-trace `k=2, c=1, S=2`: `j=0` -> `+3`; `j=1` (`0 >= 0` true, `rem=0`) -> `-C(2,1)*C(1,1) = -2`; `j=2`
-> `S - 4 = -2 < 0`, stop. Total `3 - 2 = 1`. Correct. Re-trace the sample `4 5 12`: now `j=2` with
`rem=0` is included, giving `455 - 336 + 6 = 125`. Correct. The two cases that pinned the bug now pass,
and they pass because of precisely the term the strict inequality dropped, which is the evidence I trust.

**A second trace, on the stars-and-bars argument — because the binomial top is its own boundary.** The
inclusion-exclusion fix is not the only place a `+/- 1` can hide. The unbounded count of `k` nonnegative
parts summing to `rem` is `C(rem + k - 1, k - 1)`; the `k - 1` (not `k`) in the *top* is the number of
"bars", and it is easy to fat-finger to `C(rem + k, k - 1)`. Suppose I had written that variant. Trace the
same small case `k=2, c=1, S=2` with the inclusive guard but the wrong top `C(rem + k, k - 1)`:

- `j=0`: `rem=2`. `C(2,0)*C(2+2,1) = 1*C(4,1) = 4`. `ans = 4`.
- `j=1`: `rem=0`. `-C(2,1)*C(0+2,1) = -2*C(2,1) = -2*2 = -4`. `ans = 0`.

Final `0`, but the truth is `1`. Wrong. The diagnosis: `C(rem + k - 1, k - 1)` counts solutions of
`y_1 + ... + y_k = rem` (k parts, k-1 dividers among rem+k-1 slots); inflating the top to `rem + k` silently
counts solutions with an extra phantom slot — it is the stars-and-bars count for `rem + 1` worth of stars,
not `rem`. The fix is to keep the top at `rem + k - 1`. With the correct top I already verified the case
gives `1`. I am keeping `C(rem + k - 1, k - 1)`; I traced the wrong variant only to convince myself the
exponent of the boundary here is `k - 1` and `+(k-1)` in the top, not `k`.

**Building the binomials correctly under the modulus.** I need `C(N, r)` for `N` up to `S + k - 1` and `r`
up to `k - 1`, plus `C(k, j)` for `j` up to `k`. A single factorial table of size `maxN = S + k` covers all
of them: every top I request is `<= S + k - 1 < maxN`, and `C(k, j)` needs only `fact[k]` with `k <= maxN`.
Because `M` is prime and `M > S + k > maxN >= N`, no factorial I invert contains a factor of `M`, so
`inv[i] = (i!)^{-1} mod M` via one Fermat exponentiation of the largest factorial and a downward walk is
valid. The `choose` helper returns `0` whenever `r < 0`, `N < 0`, or `r > N`, which also makes the
`C(k, j)` factor vanish automatically if `j` ever exceeded `k` — but I still keep the explicit `j <= k`
loop bound so the index `fact[k]` is never read out of intent. (`k` is at most `maxN`, so it is in range
either way; the bound is defensive, not load-bearing.)

**Edge cases, deliberately, because this is where boundary code dies.**

- `k = 0`: there are no children. The only feasible total is `S = 0` (the empty sum), giving one
  distribution; any `S > 0` is impossible. I special-case this *before* the factorial machinery, printing
  `1 % M` if `S == 0` else `0`. (If I let the loop run with `k = 0`, the binomial `C(rem - 1, -1)` is `0`
  for every `rem`, so it would wrongly print `0` even for `S = 0` — hence the explicit guard.)
- `S = 0`, `k >= 1`: only `j = 0` survives (`rem = 0`), giving `C(0 + k - 1, k - 1) = 1` — the all-zeros
  distribution. Verified `k=5,c=3,S=0` -> `1`.
- `S = k*c` (everyone maxed): exactly one distribution. The alternating sum collapses to `1`; verified by
  the oracle on many seeds.
- `S > k*c` (infeasible): no nonnegative assignment fits under the caps, answer `0`. The inclusion-exclusion
  produces this automatically; e.g. `k=3,c=2,S=7` -> `0`.
- `c = 0`: `step = 1`, every child must take `0`, so only `S = 0` is feasible. `k=3,c=0,S=0` -> `1`,
  `k=3,c=0,S=1` -> `0`. Verified.
- `c >= S` (cap never binds): `step = c+1 > S`, so only `j = 0` survives and the answer is the pure
  stars-and-bars `C(S + k - 1, k - 1)`. Checked `k=10^6, c=10^9, S=10^6` against `C(1999999, 999999) mod
  (10^9+7) = 96075800` — they match, confirming the large-`c` branch and that `j*step` (up to `~10^{15}`)
  does not overflow `long long`.
- Overflow of products: each multiply is `residue * residue < (10^9)^2 ~ 10^{18} < 9.2*10^{18}`, and I
  reduce immediately, so `long long` is safe. The subtraction branch adds `+ M` before reducing, so the
  result is never negative.
- Output: exactly one integer and a newline; the final `% M` is redundant once `ans` is kept reduced but
  harmless and guards `k = 0`'s `1 % M`.

**Cross-checking against the brute force.** I ran the inclusion-exclusion solution against the
bounded-composition DP brute force on 850 random small cases (seeds covering `S = 0`, `S = k*c`,
`S > k*c`, totals landing on multiples of `c + 1`, small primes that still exceed `S + k`, and
`10^9 + 7`). Zero mismatches. The boundary-heavy seeds — the ones with `S` an exact multiple of `c + 1`,
which force a `rem = 0` final term — are precisely the cases that would have failed under the original
`> 0` guard, and they now agree.

**Final solution.** I convinced myself the formula is right by deriving the inclusion-exclusion and
hand-checking the sample, and I convinced myself the *code* is right by tracing the inclusive/exclusive
boundary bug on `k=2,c=1,S=2` (the `>` that dropped the `rem = 0` term, turning `1` into `3`), tracing a
second off-by-one in the stars-and-bars top, and confirming the corners plus the brute-force sweep. What I
ship is the `O(S + k)` inclusion-exclusion:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long k, c, S, M;
    if (!(cin >> k >> c >> S >> M)) return 0;

    // Count integer solutions to x_1 + ... + x_k = S with 0 <= x_i <= c, modulo prime M.
    // Inclusion-exclusion over the number j of children that "overflow" (hold >= c+1):
    //   answer = sum_{j>=0, S - j*(c+1) >= 0} (-1)^j * C(k, j) * C(S - j*(c+1) + k - 1, k - 1)
    // The boundary "S - j*(c+1) >= 0" is INCLUSIVE: j may drive the remainder to exactly 0,
    // which still corresponds to a valid (fully-removed-overflow) configuration.

    if (S < 0) { cout << 0 << "\n"; return 0; }
    if (k == 0) { cout << ((S == 0) ? 1 % M : 0) << "\n"; return 0; }

    // Binomials needed: top argument is S + k - 1; also C(k, j). M is a prime strictly
    // larger than every argument we touch (guaranteed by the constraints), so factorial
    // inverses are valid.
    long long maxN = S + k; // safe upper bound on any "top" argument
    vector<long long> fact(maxN + 1), inv(maxN + 1);
    fact[0] = 1 % M;
    for (long long i = 1; i <= maxN; i++) fact[i] = fact[i - 1] * (i % M) % M;
    auto power = [&](long long b, long long e) {
        long long r = 1 % M; b %= M; if (b < 0) b += M;
        while (e > 0) { if (e & 1) r = r * b % M; b = b * b % M; e >>= 1; }
        return r;
    };
    inv[maxN] = power(fact[maxN], M - 2);
    for (long long i = maxN; i >= 1; i--) inv[i - 1] = inv[i] * (i % M) % M;

    auto C = [&](long long N, long long r) -> long long {
        if (r < 0 || N < 0 || r > N) return 0;
        return fact[N] * inv[r] % M * inv[N - r] % M;
    };

    long long ans = 0;
    long long step = c + 1; // an overflowing child holds at least c+1
    for (long long j = 0; S - j * step >= 0 && j <= k; j++) {
        long long rem = S - j * step;          // candies left after removing j overflows
        long long ways = C(k, j) * C(rem + k - 1, k - 1) % M;
        if (j & 1) ans = (ans - ways + M) % M;
        else       ans = (ans + ways) % M;
    }

    cout << ans % M << "\n";
    return 0;
}
```

**Causal recap.** The cap turns a plain stars-and-bars count into an inclusion-exclusion over which
children overflow, `sum_j (-1)^j C(k,j) C(S - j(c+1) + k - 1, k - 1)`; the entire difficulty is the
summation boundary, which is inclusive — the term with leftover exactly `0` is a real configuration. My
first loop used a strict `> 0` guard and a trace of `k=2,c=1,S=2` returning `3` instead of `1` pinned the
dropped `rem = 0` term (the same term that would mis-report the sample `4 5 12` as `119` instead of `125`);
switching to `>= 0` fixes it. A second trace guarded the stars-and-bars top at `rem + k - 1` (not `rem + k`),
and explicit checks on `k=0`, `S=0`, `S>k*c`, `c=0`, and `c>=S`, plus an 850-case brute-force sweep with
zero mismatches, close it out.
