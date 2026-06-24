**Problem.** A rod has `n` unit-segments `0..n-1` with integer values `v[i]` (possibly negative). Cut the rod into contiguous billets that tile all of `[0, n)`, each billet's length in `[L, R]` (`1 <= L <= R <= 50`). A billet over segments `[j, i)` costs `K + |v[j] + ... + v[i-1]|`. Minimize the total cost; print `-1` if no legal tiling exists; `n = 0` costs `0`. Read `n K L R` then the `n` values from stdin, print the answer.

**Why a heuristic fails.** Fixed-length cutting (always `R`, always `L`) or greedy "grow while `|sum|` shrinks" is wrong on two counts: with negative `v` the cost-minimizing split is the one that keeps each billet's running sum near zero, which does not align with any fixed length; and a fixed rule produces infeasible leftovers (e.g. `L = R = 2`, `n = 3`) with no way to report `-1`. A DP that scans the full legal window handles both.

**Key idea — linear partition DP on prefix cuts.** Use prefix sums `S[i] = v[0] + ... + v[i-1]` so a billet `[j, i)` has imbalance `S[i] - S[j]` in `O(1)`. Let `dp[i]` be the minimum cost to tile the prefix `[0, i)`, with `dp[i] = +inf` when untileable. Base `dp[0] = 0`. The last billet `[j, i)` has length `i - j`, which must satisfy `L <= i - j <= R`. Solving the double inequality for the predecessor index:

- `i - j <= R`  =>  `j >= i - R`
- `i - j >= L`  =>  `j <= i - L`

so `j` ranges over the **inclusive** integer window `[max(0, i - R), i - L]`, and

```
dp[i] = min over j in [max(0,i-R), i-L] of ( dp[j] + K + |S[i] - S[j]| )   if dp[j] finite.
```

The answer is `dp[n]`, or `-1` if `dp[n]` is still `+inf`. With `R <= 50` the window is at most 50 wide, so this is `O(n * R)`.

**Correctness.** Every tiling of `[0, n)` ends with a last billet `[j, n)` of legal length, and `dp[j]` optimally tiles the rest by induction; the transition considers exactly the legal last billets and adds their cost, so `dp[i]` is the true optimum for `[0, i)`. The `dp[j] >= INF` guard prevents building a finite cost on an untileable prefix, so `dp[n] = inf` iff no legal tiling exists, which we map to `-1`.

**Pitfalls.**
1. *Off-by-one window (the crux).* The constraint is on *length* but the transition is on the *predecessor* `j`. The inclusive high end is `j = i - L` (shortest legal billet, length `L`), not `i - L + 1`. Writing `jhi = i - L + 1` admits a billet of length `L - 1` — one segment too short. It is wrong on a large fraction of inputs: e.g. `n=1, K=37, L=R=2, v=[9]` should be `-1` (one segment can't form a length-2 billet) but the buggy window outputs `46`. Tracing that case pins the high end to `i - L`.
2. *Infeasibility / unreachable predecessor.* Skip `j` with `dp[j] = INF` (`if (dp[j] >= INF) continue;`); otherwise an impossible prefix leaks a finite-looking cost and you print a giant number instead of `-1`. Trace `n=3, L=R=2, v=[1,1,1]` -> `-1`.
3. *Overflow.* Total cost is bounded by `n * (K + R*max|v|) ~ 2*10^5 * 5.1*10^10 ~ 1.0*10^16`, which fits `long long` but overflows 32-bit by orders of magnitude — use `long long` for `S`, `dp`, and each billet cost. Pick `INF = 4e18` so even an unguarded `INF + cost` cannot overflow `long long`, keeping the `>= INF` boundary test valid.

**Edge cases (all fall out of the recurrence + window):** `n = 0` -> `dp[0] = 0` -> `0`; `n < L` -> empty window -> `dp[n] = inf` -> `-1`; `L = R` forces the window to a single `j`; single segment with `L = 1` -> `K + |v[0]|`; all-equal / all-negative `v` handled by `| . |`.

**Complexity.** `O(n * (R - L + 1)) = O(n * R)` time (`<= 10^7`), `O(n)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long K;
    int L, R;
    if (!(cin >> n >> K >> L >> R)) return 0;
    vector<long long> v(n);
    for (auto &x : v) cin >> x;

    // prefix sums S[0..n], S[i] = v[0] + ... + v[i-1]
    vector<long long> S(n + 1, 0);
    for (int i = 0; i < n; i++) S[i + 1] = S[i] + v[i];

    const long long INF = (long long)4e18;
    // dp[i] = min cost to partition the first i unit-segments, i.e. the half-open
    // range [0, i), into valid billets. dp[0] = 0 (nothing cut yet).
    // A billet covering segments [j, i) has length (i - j), which must satisfy
    // L <= i - j <= R. Solving for j: j in [i - R, i - L], and also j >= 0.
    // The billet's cost is K + |S[i] - S[j]| (a setup fee plus the imbalance).
    vector<long long> dp(n + 1, INF);
    dp[0] = 0;
    for (int i = 1; i <= n; i++) {
        int jlo = max(0, i - R);   // longest allowed billet, length R
        int jhi = i - L;           // shortest allowed billet, length L
        for (int j = jlo; j <= jhi; j++) {
            if (dp[j] >= INF) continue;
            long long seg = S[i] - S[j];
            long long cost = K + llabs(seg);
            if (dp[j] + cost < dp[i]) dp[i] = dp[j] + cost;
        }
    }

    if (dp[n] >= INF) cout << -1 << "\n";
    else cout << dp[n] << "\n";
    return 0;
}
```
