**Problem.** Count the multisets of size `k` over `n` distinct flavours where each flavour is used at
most `c` times, modulo `p = 1 000 000 007`. Equivalently, count integer tuples `(x_1, ..., x_n)` with
`0 <= x_i <= c` and `sum x_i = k`. Read `n k c` from stdin; print the count mod `p`. Constraints:
`n, k, c` up to `2*10^6` with `n + k <= 4*10^6`, 1 s.

**Why the direct DP is too slow.** Convolving flavour by flavour (`dp[s]` = ways to reach running sum
`s`, spreading each new flavour over the window `0..c`) is `O(n*k)` ~ `4*10^12` at the limits. Correct,
but only usable as an oracle.

**Key idea — inclusion-exclusion over capped flavours.** Without the cap the count is stars and bars
`C(n + k - 1, n - 1)`. Subtract the tuples where some flavour exceeds `c`. Forcing a chosen flavour to
use at least `c + 1` candies substitutes `x_i = y_i + (c + 1)` and removes `c + 1` from the budget; a
forced set of size `j` costs `j*(c+1)` and is chosen `C(n, j)` ways with sign `(-1)^j`:

  `answer = sum_{j>=0} (-1)^j * C(n, j) * C(n + (k - j*(c+1)) - 1, n - 1).`

Stop the sum the first time the residual budget `r_j = k - j*(c+1)` is negative (it only decreases) or
`j > n`. Precompute factorials and inverse factorials up to `n + k` (`O(n + k)`); each term is `O(1)`,
so the whole solution is linear in `n + k`.

**Pitfalls to get right.**
1. *Empty-bin off-by-one (the double-/under-count trap).* The factor `C(n + r - 1, n - 1)` is the
   number of nonnegative length-`n` tuples summing to `r`, but the stars-and-bars formula only holds
   for `n >= 1`. For `n = 0` there is exactly one tuple (the empty one) when `r = 0` and none
   otherwise; the binomial `C(r - 1, -1)` returns `0`, silently dropping the empty box. A trace of
   `n = k = c = 0` (true answer `1`, formula gives `0`) exposes it; fix with an explicit `bars(m, r)`
   helper that returns `(r == 0)` when `m == 0`.
2. *Stopping rule.* Break exactly when `r_j < 0` or `j > n`; including a term with negative residual,
   or omitting the `j > n` guard, double-counts or reads `fact` out of range.
3. *Overflow.* Reduced factors are `< 2^30`, so a two-way product is `~2^60` (fits) but a three-way
   product or a refactor can exceed `2^63`. Route every modular product through `__int128`.

**Edge cases.** `k = 0` -> `1` (unique empty box, including `n = 0`); `c = 0` -> `1` iff `k = 0` else
`0`; `k > n*c` -> `0` (loop runs only `j <= n`, stays fast); `c >= k` -> only `j = 0` survives, giving
the plain `C(n + k - 1, n - 1)`; `k` a multiple of `c + 1` exercises `bars(n, 0) = 1`.

**Complexity.** `O(n + k)` time and memory for the factorial tables; the inclusion-exclusion loop runs
at most `min(n, k/(c+1)) + 1` iterations, each `O(1)`.

**Code.**

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
