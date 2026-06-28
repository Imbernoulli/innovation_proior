**Problem.** Given two integer arrays `A` (length `n`) and `B` (length `m`), values possibly negative, pick a **non-empty** subsequence of each of the **same length** `k >= 1`, keep the original order within each array, align them position by position, and maximize the dot product `A[i_1]*B[j_1] + ... + A[i_k]*B[j_k]` with `i_1 < ... < i_k` and `j_1 < ... < j_k`. The empty pairing is disallowed, so the answer may be negative. Read `n m`, then `A`, then `B` from stdin; print the maximum dot product.

**Why the tempting greedies are wrong.**

- *Single best product.* Since a length-1 pairing is always legal, one is tempted to return `max_{i,j} A[i]*B[j]`. This ignores that pairs **stack**: on `A = [2, 1, -2]`, `B = [3, 0, -1]` the best single product is `A[0]*B[0] = 6`, but the length-2 alignment `A[0]*B[0] + A[2]*B[2] = 6 + 2 = 8` is strictly better. Discarded.
- *Sort and pair largest magnitudes.* After the first failure one might sort both arrays and pair the biggest with the biggest. This ignores **order-preservation**: on `A = [3, -5]`, `B = [-5, 3]` it claims `3*3 + (-5)*(-5) = 34`, but realizing both pairs would cross the index order; the only legal length-2 alignment scores `3*(-5) + (-5)*3 = -30`, and the true optimum is the length-1 pairing `(-5)*(-5) = 25`. Discarded.

Both fail for reasons a greedy cannot patch without rebuilding the DP, so I ship the provable alignment DP.

**Key idea — O(nm) alignment DP.** Let `dp[i][j]` be the best dot product of a **non-empty** aligned pairing using a subsequence of `A[0..i-1]` and of `B[0..j-1]`. The last cell's decision is over `A[i-1]`, `B[j-1]`:

```
dp[i][j] = max( A[i-1]*B[j-1],                  // start a brand-new length-1 pairing
                dp[i-1][j-1] + A[i-1]*B[j-1],   // extend a real pairing (only if dp[i-1][j-1] exists)
                dp[i-1][j],                     // drop A[i-1]
                dp[i][j-1] )                    // drop B[j-1]
```

Borders `dp[*][0]`, `dp[0][*]` are a sentinel `NEG` meaning "no non-empty pairing yet". The standalone `A[i-1]*B[j-1]` term guarantees every interior cell has a real value, so `dp[n][m]` is always a valid answer.

**Two pitfalls to get right.**
1. *Non-empty enforcement.* Keep the standalone product as a first-class candidate and only add the product to `dp[i-1][j-1]` when it is a real pairing (`!= NEG`); otherwise you either pretend an empty pairing exists or you add into the sentinel. There is deliberately **no** `max(..., 0)` — the empty pairing is disallowed, so forced-negative inputs like `A=[-5], B=[3]` must return `-15`.
2. *Sentinel safety.* `NEG = LLONG_MIN/4` is only read inside `max` or behind the `!= NEG` guard; the product is never added to it, so no underflow. Products reach `10^6`, sums `5*10^8` — well inside `long long`.

**Edge cases (handled by the recurrence):** `n=1` or `m=1` caps the length at 1; all-negative arrays yield large positives by stacking two-negative products; zeros are just ordinary candidate values; mismatched lengths cap at `min(n,m)`.

**Complexity.** `O(nm)` time, `O(nm)` space (reducible to `O(m)`), about `250000` cells at the maximum size — roughly 3 ms, far under the 1-second limit.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<long long> A(n), B(m);
    for (auto &x : A) cin >> x;
    for (auto &x : B) cin >> x;

    // dp[i][j] = best dot product of a NON-EMPTY equal-length pairing using
    // a subsequence of A[0..i-1] and of B[0..j-1]. NEG sentinel = "no non-empty
    // pairing exists yet for this prefix pair".
    const long long NEG = LLONG_MIN / 4;
    vector<vector<long long>> dp(n + 1, vector<long long>(m + 1, NEG));

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            long long prod = A[i - 1] * B[j - 1];
            // pair A[i-1] with B[j-1], either as the only pair or extending dp[i-1][j-1]
            long long best = prod;                       // start a brand-new pairing here
            if (dp[i - 1][j - 1] != NEG)
                best = max(best, dp[i - 1][j - 1] + prod); // extend an existing pairing
            // or drop A[i-1] / drop B[j-1]
            best = max(best, dp[i - 1][j]);
            best = max(best, dp[i][j - 1]);
            dp[i][j] = best;
        }
    }

    cout << dp[n][m] << "\n";
    return 0;
}
```
