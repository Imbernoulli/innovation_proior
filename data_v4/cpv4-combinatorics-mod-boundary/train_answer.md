**Problem.** Distribute `S` identical candies among `k` distinguishable children, each child receiving between `0` and `c` inclusive; count the distributions — the integer solutions of `x_1 + ... + x_k = S` with `0 <= x_i <= c` — modulo a prime `M` (with `M > S + k`). Read `k c S M` from stdin, print the count mod `M`. Constraints: `k, S <= 10^6`, `c <= 10^9`.

**Key idea — inclusion-exclusion over the cap.** The unbounded count of `k` nonnegative parts summing to `T` is stars-and-bars `C(T + k - 1, k - 1)`. A child violates its cap exactly when it holds `>= c + 1`; subtracting `c + 1` from each of `j` forced-overflow children removes their caps and lowers the total by `j*(c+1)`. Inclusion-exclusion over which children overflow gives

```
answer = sum_j (-1)^j * C(k, j) * C(S - j*(c+1) + k - 1, k - 1)   (mod M),
```

summed over every `j` with `S - j*(c+1) >= 0` and `j <= k`. Precompute factorials and inverse factorials up to `S + k` (valid because `M` is prime and exceeds `S + k`), then evaluate the at-most-`k+1`-term alternating sum. `O(S + k)` time.

**Pitfalls.**
1. *Inclusive boundary of the sum (the off-by-one).* The stopping test is `S - j*(c+1) >= 0`, **not** `> 0`. The term where the leftover is exactly `0` — every overflow exactly consuming the total — is a real configuration. A strict `> 0` drops it: on `k=2, c=1, S=2` it returns `3` instead of `1` (the missing `j=1`, `rem=0` term is `-2`); on the sample `k=4, c=5, S=12` it returns `119` instead of `125` (dropping the `j=2`, `rem=0` term `+6`).
2. *Stars-and-bars top.* The unbounded count is `C(rem + k - 1, k - 1)`; using `rem + k` in the top silently counts one extra phantom slot and corrupts every term.
3. *`k = 0` corner.* No children means the only feasible total is `S = 0` (answer `1`), else `0`. The factorial formula would mis-handle this (it forms `C(-1, -1)`), so special-case it before the loop.
4. *Overflow.* Reduce after every multiply; products are `< 10^{18}` and fit `long long`. `j*(c+1)` reaches `~10^{15}`, also fine in `long long`. Subtract with `+ M` before reducing so results stay nonnegative.

**Edge cases.** `S = 0` -> `1` (all zeros); `S = k*c` -> `1` (everyone maxed); `S > k*c` -> `0` (infeasible, handled automatically by the alternating sum); `c = 0` -> `1` iff `S = 0` else `0`; `c >= S` -> only the `j = 0` term survives, giving the pure stars-and-bars `C(S + k - 1, k - 1)`; `k = 0` -> `1` iff `S = 0` (special-cased).

**Complexity.** `O(S + k)` time and memory for the factorial table; `O(k)` for the summation.

**Code.**

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
