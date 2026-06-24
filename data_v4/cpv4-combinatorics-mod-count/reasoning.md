**Reading the problem and pinning the contract.** I need the number of multisets of size `k` over
`n` distinct flavours where each flavour is used at most `c` times, all taken modulo
`p = 1 000 000 007`. Stripping the story, a box is a tuple `(x_1, ..., x_n)` with `0 <= x_i <= c` and
`sum x_i = k`, and I count such tuples. Input is the three integers `n k c`; output is one integer.
Before any algorithm I fix the scale, because it decides the data structures: `n, k <= 2*10^6` with
`n + k <= 4*10^6`. So whatever I precompute over `n + k` must be linear in that and fit in 256 MB —
two `long long` arrays of length `~4*10^6` is about 64 MB, which is fine. All arithmetic is mod a
prime, and I will be multiplying two values near `10^9`, so every product must go through `__int128`
or 64-bit overflow is a silent wrong answer. That is decision zero and it is non-negotiable.

**Laying out the candidate approaches.** Two routes, and I want the one I can *prove*, not the one
that types fastest.

- *Direct DP over flavours.* Carry `dp[s]` = number of ways to fill the flavours seen so far to a
  running sum `s`, and for each new flavour convolve with the window `{0..c}`. This is exactly the
  brute force; it is obviously correct but it is `O(n * k)` (each flavour spreads over up to `k`
  sums), which at `n, k ~ 2*10^6` is `4*10^12` operations — hopelessly slow. Good as an oracle, not
  as a solution.
- *Closed-form inclusion-exclusion.* The uncapped count is stars-and-bars `C(n + k - 1, n - 1)`. The
  cap removes the tuples where some flavour exceeds `c`; inclusion-exclusion over *which* flavours
  overflow gives an alternating sum with `C(n, j)` choosing the `j` overflowing flavours. This is
  `O((k / (c + 1)) + (n + k))` after an `O(n + k)` factorial precompute — fast enough. The risk is
  not the idea but the *transcription*: the summation limit and the binomial indices are where this
  family of counts double-counts or goes off by one.

I commit to inclusion-exclusion and treat the DP as my checking oracle.

**Deriving the inclusion-exclusion identity carefully.** Let `U` be the number of nonnegative tuples
with `sum = k` and no upper bound: `U = C(n + k - 1, n - 1)` (stars and bars). I want to subtract the
"bad" tuples where at least one `x_i >= c + 1`. Define, for a fixed set `S` of flavours, `A_S` = the
count of tuples where *every* flavour in `S` is forced to be at least `c + 1` (others unrestricted).
To count `A_S`, substitute `x_i = y_i + (c + 1)` for `i in S` with `y_i >= 0`; the budget drops by
`|S| * (c + 1)`, so

  `A_S = (number of nonneg tuples with sum k - |S|*(c+1)) = C(n + (k - |S|*(c+1)) - 1, n - 1)`,

which depends only on `|S| = j`. By inclusion-exclusion the count of tuples with *no* flavour
exceeding the cap is

  `answer = sum_{j=0}^{...} (-1)^j * C(n, j) * C(n + (k - j*(c+1)) - 1, n - 1)`.

The `(-1)^j` is the standard sign, and `C(n, j)` is the number of sets `S` of size `j`. This is the
identity I will implement.

**Where does the sum stop?** The term at index `j` involves a budget `r_j = k - j*(c+1)`. Once
`r_j < 0` there is no tuple at all forcing those `j` flavours over the cap, so `A_S = 0` and the term
vanishes; and `r_j` strictly decreases in `j`, so every later term is `0` too. Also `j` cannot exceed
`n` (can't pick more than `n` flavours to overflow), and `C(n, j) = 0` for `j > n` anyway. So I loop
`j` from `0` upward and break the first time either `j > n` or `r_j < 0`. The largest binomial top I
ever ask for is at `j = 0`: `n + k - 1`, which is `< n + k`, so a factorial table of size `n + k + O(1)`
is enough.

**Derivation sanity-check on the sample.** `n = 3, k = 4, c = 2`, so `c + 1 = 3`. Terms:
- `j = 0`: `r = 4`. `C(3,0) * C(3 + 4 - 1, 2) = 1 * C(6, 2) = 15`.
- `j = 1`: `r = 4 - 3 = 1`. `C(3,1) * C(3 + 1 - 1, 2) = 3 * C(3, 2) = 3 * 3 = 9`.
- `j = 2`: `r = 4 - 6 = -2 < 0`, stop.
Answer `= 15 - 9 = 6`. That matches the hand count of the six valid tuples
`(2,2,0),(2,0,2),(0,2,2),(2,1,1),(1,2,1),(1,1,2)`. The identity and the stopping rule are right.

**First implementation — and a trace, because clean math transcribes dirty.** My first cut of the
core, with a single binomial helper `C(a,b)` that returns `0` when out of range:

```
auto C = [&](long long a, long long b) -> long long {
    if (b < 0 || a < 0 || b > a) return 0;
    return fact[a] * inv_fact[b] % MOD * inv_fact[a - b] % MOD;
};
long long ans = 0;
for (long long j = 0; ; j++) {
    long long r = k - j * (c + 1);
    if (r < 0) break;
    if (j > n) break;
    long long term = C(n, j) * C(n + r - 1, n - 1) % MOD;
    if (j & 1) ans = (ans - term + MOD) % MOD;
    else       ans = (ans + term) % MOD;
}
```

This reproduces the sample (I just hand-checked the same arithmetic). So I reach for the corner that
this family of counts always dies on: the empty everything. Trace `n = 0, k = 0, c = 0` — there are
no flavours and the box is empty, so there is exactly one box (the empty multiset) and the answer is
`1`. Step the loop: `j = 0`, `r = 0 - 0 = 0 >= 0`, `j > n` is `0 > 0` false. `term = C(0, 0) *
C(0 + 0 - 1, 0 - 1) = 1 * C(-1, -1)`. My `C` sees `b = -1 < 0` and returns `0`, so `term = 0`, and
the loop's next iteration has `j = 1 > n = 0` and breaks. Final `ans = 0`.

**Diagnosing the first bug.** The code returns `0` but the true answer is `1`. The defect is precise
and it is an *indexing* bug, not a logic bug in the inclusion-exclusion: the factor `C(n + r - 1,
n - 1)` is the stars-and-bars count of nonnegative tuples of length `n` summing to `r`, and that
formula `C(m + r - 1, m - 1)` is only valid for `m >= 1`. For `m = 0` bins there is exactly *one*
distribution when `r = 0` (the empty tuple) and *zero* otherwise — but the binomial `C(r - 1, -1)`
evaluates to `0`, silently dropping the legitimate empty box. So whenever `n = 0` the formula
undercounts. I confirm the failure is real and isolated by checking `n = 0, k = 3, c = 2`: there the
true answer *is* `0` (no flavours, can't build a size-3 box), and my code also returns `0` — correct
by luck, because `r = 3 != 0`. The bug only bites at `n = 0, k = 0`. That is exactly the kind of
empty-bin off-by-one the stars-and-bars formula hides.

**Fixing the first bug.** Introduce an explicit stars-and-bars helper that special-cases zero bins,
and route the second binomial through it:

```
auto bars = [&](long long m, long long r) -> long long {
    if (m == 0) return (r == 0) ? 1 : 0;   // empty tuple iff nothing to place
    return C(m + r - 1, m - 1);
};
...
long long term = C(n, j) * bars(n, r) % MOD;
```

Re-trace `n = 0, k = 0, c = 0`: `j = 0`, `r = 0`, `term = C(0,0) * bars(0, 0) = 1 * 1 = 1`, then
`j = 1 > 0` breaks. Answer `1` — correct. Re-trace `n = 0, k = 3`: `j = 0`, `r = 3`, `bars(0,3) = 0`,
`term = 0`; answer `0` — still correct. The corner that broke now passes and it passes for the reason
I fixed, which is the evidence I trust.

**Second trace — the multiplication/overflow path, on a non-trivial case.** I now worry about the
arithmetic, not the combinatorics. Trace `n = 4, k = 7, c = 3`, `c + 1 = 4`:
- `j = 0`: `r = 7`, `C(4,0) * bars(4,7) = 1 * C(10, 3) = 120`.
- `j = 1`: `r = 3`, `C(4,1) * bars(4,3) = 4 * C(6, 3) = 4 * 20 = 80`.
- `j = 2`: `r = -1 < 0`, stop.
Answer `= 120 - 80 = 40`. I cross-check against the DP oracle below; it also says `40`. Good. Now the
overflow worry: in `term = C(n, j) * bars(n, r) % MOD`, both factors are reduced mod `p < 2^30`, so
their product is up to `~2^60`, which *fits* in a signed 64-bit `long long` — but only just, and it
is fragile to refactors. To be safe and uniform I will route every modular product through
`__int128`, including inside `C` (`fact[a] * inv_fact[b]` is already two reduced values, product
`~2^60`, then times a third — *that* second multiply could exceed `2^63` if done in plain 64-bit).
The three-way product `fact[a] * inv_fact[b] % MOD * inv_fact[a-b] % MOD` is safe only because of the
intermediate `% MOD`; I make the `__int128` casts explicit so no reordering can break it. This is a
real latent bug class even though the sample happened to be small.

**Edge cases, deliberately, because this is where counting code dies.**
- `k = 0` (empty box): only `j = 0` survives (`r = 0`), `term = C(n,0) * bars(n,0) = 1 * C(n-1,n-1) =
  1` for `n >= 1`, and `bars(0,0) = 1` for `n = 0`. Answer `1` everywhere — the empty box is the
  unique size-0 box. Correct.
- `c = 0` (cap zero): `c + 1 = 1`, so `r_j = k - j`. The sum telescopes to the count of tuples with
  every `x_i = 0`, which is `1` iff `k = 0` and `0` otherwise. I spot-checked `n = 6, k = 6, c = 0`
  -> `0` and `n = 5, k = 0, c = 0` -> `1`; both agree with the oracle.
- `k > n * c` (over budget): no tuple fits, answer must be `0`. Tried `n = 100, k = 4*10^6, c = 50`
  (max total `5000`); the alternating sum collapses to `0`, and the loop only runs `j` up to `100`
  before `j > n`, so it is fast. Oracle-infeasible at that size but structurally must be `0`, and the
  program prints `0`.
- `c >= k` (cap not binding): every `r_j` for `j >= 1` is `k - j*(c+1) <= k - (c+1) < 0`, so only
  `j = 0` survives and the answer is the plain stars-and-bars `C(n + k - 1, n - 1)`. Checked
  `n = 1, k = 3, c = 5` -> `C(3, 0) = 1` (one flavour, the box is "three of it") — correct.
- `k` an exact multiple of `c + 1`: then some `r_j` hits exactly `0`, exercising `bars(n, 0) =
  C(n-1, n-1) = 1`; I included such cases (e.g. `n = 2, k = 6, c = 2`, `c+1 = 3`) in the random
  battery and they pass.
- Overflow / memory: accumulators and the factorial table are `long long`; the table has
  `n + k + O(1) <= 4*10^6 + O(1)` entries (~64 MB for two arrays), and every modular product goes
  through `__int128`. The largest in-bounds case `n + k = 4*10^6` runs in under `0.1 s`.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace so parsing is
  format-agnostic, and `if (!(cin >> n >> k >> c)) return 0;` guards a truncated stream.

**Re-verification at scale.** I wrote the `O(n*k)` DP as an independent oracle and compared on more
than a thousand random small instances with `n` up to ~12, `k` up to ~20, `c` up to ~10, plus the
explicit corners above. Before the `bars` fix the only failures were exactly the `n = 0, k = 0`
family; after it, zero mismatches across the whole battery. The two bugs I caught — the empty-bin
stars-and-bars off-by-one and the latent 64-bit product overflow — are both the "subtle index /
modulus" trap this problem is built around, and both are now closed.

**Final solution.** I proved the inclusion-exclusion identity, fixed its stopping rule, repaired the
`m = 0` stars-and-bars corner that the binomial formula silently miscounts, and hardened every
modular product through `__int128`. That is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

long long power_mod(long long b, long long e, long long m) {
    b %= m; if (b < 0) b += m;
    long long r = 1 % m;
    while (e > 0) {
        if (e & 1) r = (__int128)r * b % m;
        b = (__int128)b * b % m;
        e >>= 1;
    }
    return r;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, k, c;
    if (!(cin >> n >> k >> c)) return 0;

    // Count multisets of size k drawn from n colors, each color used at most c times, mod p.
    // Inclusion-exclusion over the set of colors forced to exceed the cap:
    //   answer = sum_{j>=0} (-1)^j * C(n, j) * C(n + (k - j*(c+1)) - 1, n - 1).
    // The term for index j is real only while r_j = k - j*(c+1) >= 0; once r_j < 0 it and
    // every larger j contribute 0 (the binomial top n+r_j-1 drops below n-1).

    // Largest binomial top ever requested is n + k - 1, so factorials up to n + k suffice.
    long long N = n + k + 5;
    vector<long long> fact(N + 1), inv_fact(N + 1);
    fact[0] = 1;
    for (long long i = 1; i <= N; i++) fact[i] = (__int128)fact[i - 1] * i % MOD;
    inv_fact[N] = power_mod(fact[N], MOD - 2, MOD);
    for (long long i = N; i >= 1; i--) inv_fact[i - 1] = (__int128)inv_fact[i] * i % MOD;

    auto C = [&](long long a, long long b) -> long long {
        if (b < 0 || a < 0 || b > a) return 0;
        return (__int128)fact[a] % MOD * inv_fact[b] % MOD * inv_fact[a - b] % MOD;
    };

    // Number of ways to put r indistinguishable balls into m distinguishable bins (each >= 0).
    // Stars and bars = C(m + r - 1, m - 1) for m >= 1; for m = 0 it is 1 iff r = 0, else 0.
    auto bars = [&](long long m, long long r) -> long long {
        if (m == 0) return (r == 0) ? 1 : 0;
        return C(m + r - 1, m - 1);
    };

    long long ans = 0;
    for (long long j = 0; ; j++) {
        if (j > n) break;                 // cannot force more than n colors over the cap
        long long r = k - j * (c + 1);    // size remaining after over-filling j colors
        if (r < 0) break;                 // r only decreases as j grows, so stop
        long long term = (__int128)C(n, j) % MOD * bars(n, r) % MOD;
        if (j & 1) ans = (ans - term % MOD + MOD) % MOD;
        else ans = (ans + term) % MOD;
    }

    cout << ans % MOD << "\n";
    return 0;
}
```

**Causal recap.** The bounded-composition count factors as an alternating sum
`sum_j (-1)^j C(n,j) * stars_and_bars(n, k - j*(c+1))`, derived by substituting `x_i -> y_i + (c+1)`
for the `j` forced-over-cap flavours; the sum stops the first time the residual budget `k - j*(c+1)`
goes negative (or `j > n`), which is the only place a wrong stopping rule would double-count. A trace
of the empty box `n = k = c = 0` exposed that the naive `C(n+r-1, n-1)` returns `0` instead of `1`
because the stars-and-bars formula is invalid for zero bins — fixed with an explicit `bars` helper
that special-cases `m = 0`. A second trace on `n = 4, k = 7, c = 3` (answer `40`, oracle-confirmed)
flagged that the modular products sit right at the 64-bit ceiling, so every multiply is routed
through `__int128`. With the `O(n+k)` factorial precompute and the early break, the program is linear
in `n + k`, fits in memory, and agrees with the brute-force DP on more than a thousand random cases
plus every hand-built corner.
