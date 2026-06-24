**Problem.** A van of volume `W` is to be loaded from `n` parcels; parcel `i` has volume `w[i]` and payout `v[i]` and can be loaded at most once. Choose a subset with total volume `<= W` maximizing total payout, and print that payout. Read `n`, `W`, then the `n` pairs from stdin. Constraints: `n <= 1000`, `W <= 10^5`, `0 <= w[i] <= 10^5`, `0 <= v[i] <= 10^9`.

**Why the obvious greedy is wrong.** "Sort by value density `v/w` and load the densest that fit" is optimal for the *fractional* knapsack but not for whole parcels. On `W = 4` with parcels `(1, 2)` and `(4, 7)`, density picks `(1, 2)` first (density `2.0` vs `1.75`), leaving only volume `3`, so the `(4, 7)` parcel no longer fits and greedy scores `2`; loading `(4, 7)` alone scores `7`. An indivisible parcel can fill the van better than several dense crumbs. Greedy is discarded.

**Key idea — capacity DP.** Let `dp[c]` be the best payout using volume at most `c`. Start `dp[c] = 0` for all `c` (empty load). Fold parcels in one at a time; for a new parcel `(w, v)`,

- `dp[c] = max(dp[c], dp[c - w] + v)` for every budget `c >= w`.

The answer is `dp[W]`. The `dp[c - w]` on the right must be the value *before* this parcel was folded in, so the parcel is used at most once.

**Correctness.** By induction on the number of parcels folded, `dp[c]` equals the maximum payout over all subsets of the processed parcels with total volume `<= c`: either the new parcel is excluded (`dp[c]` unchanged) or included (`dp[c - w] + v`, optimal for the rest within the reduced budget). `dp` is monotone non-decreasing in `c`, so `dp[W]` is the best over all volumes up to `W`.

**Pitfalls (the two that sink this).**
1. *Item reuse from the wrong sweep.* Iterate the budget `c` from `W` **down to** `w`. An ascending sweep lets `dp[c - w]` already include the current parcel, turning 0/1 into unbounded knapsack — one parcel `(2, 5)` in budget `6` then returns `15` (three copies) instead of `5`.
2. *Silent int overflow.* With `v[i]` up to `10^9` and up to `1000` parcels, the optimal total reaches `~10^{12}`, far past `2^31 - 1 ≈ 2.1 * 10^9`. A 32-bit `dp[c - w] + v` wraps to a negative number that the `max` discards, yielding a too-small wrong answer with no crash. The whole value path — `dp`, `v`, and the candidate sum — must be `long long`. (The sample answer `3300000000` already exceeds `2^31 - 1`, so it doubles as an overflow tripwire.)

**Edge cases.** `n = 0` -> `0` (loop never runs). `W = 0` -> only zero-volume positive parcels can be loaded; a `(0, v)` parcel sets `dp[0] = v`. Every `w[i] > W` -> the inner loop and the `w > W` guard skip them, answer `0`. Zero-volume parcels are loaded exactly once per budget (read index equals write index, written from its own pre-parcel value). All payouts are `>= 0`, so the base case `0` needs no negative sentinel.

**Complexity.** `O(n * W)` time (up to `10^8` tight integer operations, ~0.05 s here), `O(W)` extra space (~0.8 MB for the `long long` table).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long W;
    if (!(cin >> n >> W)) return 0;

    vector<long long> wt(n), val(n);
    for (int i = 0; i < n; i++) cin >> wt[i] >> val[i];

    // dp[c] = maximum total value achievable using a capacity budget of exactly-at-most c.
    // 0/1 knapsack: each item used at most once -> iterate capacity downward.
    vector<long long> dp(W + 1, 0);
    for (int i = 0; i < n; i++) {
        long long w = wt[i], v = val[i];
        if (w > W) continue;                       // item never fits
        for (long long c = W; c >= w; c--) {
            long long cand = dp[c - w] + v;        // both operands long long -> no overflow
            if (cand > dp[c]) dp[c] = cand;
        }
    }

    cout << dp[W] << "\n";
    return 0;
}
```
