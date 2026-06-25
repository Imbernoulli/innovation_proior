**Problem.** A relay satellite has one battery of `W` energy units. Experiment `i` costs `e[i]` energy and returns `v[i]` science value, and each experiment exists in a single copy, so it can run **at most once**. Choose a subset with total energy `<= W` maximizing total value (running nothing gives `0`). Read `n`, `W`, then the `n` pairs from stdin; print the maximum value. Constraints: `n <= 2000`, `W <= 2*10^5`, `e[i] <= 10^5`, `v[i] <= 10^9`.

**Key idea.** This is 0/1 knapsack. Use the space-saving 1D rolling array `dp[c]` = best value within energy budget `c`, updated item by item:

`dp[c] = max(dp[c], dp[c - e[i]] + v[i])`.

The whole game is the **capacity loop direction**. Sweeping `c` **descending** from `W` down to `e[i]` reads `dp[c - e[i]]` while it still holds the *previous* item's row, so each experiment is used at most once — correct for this variant. `O(n*W)` time, `O(W)` memory.

**Pitfalls.**
1. *Wrong loop direction (the headline trap).* The "standard" 1D knapsack snippet that sweeps capacity **ascending** (`for c = e[i] .. W`) is the *unbounded* knapsack: `dp[c - e[i]]` was already updated this pass, so an item gets reused. It compiles, runs fast, and looks canonical, but it answers the wrong question. On the sample it returns `26` and its fingerprint is `dp[9] = 24 = 3*8` after a single `(3,8)` item — one instrument run three times. Because `e[i]` can be far below `W`, the deviation is large, not a corner. Always sweep descending for at-most-once.
2. *Overflow.* Up to `2000` items of value `10^9` give totals near `2*10^12`, past 32 bits — use `long long` for `dp`.
3. *The feasibility guard.* Skip an item only when `e[i] > W` (strict). Using `e[i] >= W` would wrongly drop an experiment that exactly fills the battery (e.g. `e = W`, which is feasible and should be taken).

**Edge cases.** `W = 0` -> every item skipped -> `0`; items heavier than `W` skipped without polluting `dp`; `v[i] = 0` items never preferred spuriously; single fitting item taken via the `c >= e[i]` bound and the strict guard. `dp` starts at `0` and only increases, so no underflow; the array is `~1.6` MB.

**Complexity.** `O(n*W)` time (`<= 4*10^8`, ~0.2 s here), `O(W)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long W;
    if (!(cin >> n >> W)) return 0;          // no data -> nothing scheduled
    vector<long long> e(n), v(n);            // e[i] = energy cost, v[i] = science value
    for (int i = 0; i < n; i++) cin >> e[i] >> v[i];

    // 0/1 knapsack: each experiment scheduled at most once, total energy <= W.
    // dp[c] = best total value using energy budget exactly-bounded-by c.
    vector<long long> dp(W + 1, 0);
    for (int i = 0; i < n; i++) {
        if (e[i] > W) continue;              // never fits, skip to avoid touching dp out of range
        // DESCENDING capacity: each item contributes to a strictly smaller earlier state,
        // so within this i the item is used at most once.
        for (long long c = W; c >= e[i]; c--) {
            long long cand = dp[c - e[i]] + v[i];
            if (cand > dp[c]) dp[c] = cand;
        }
    }

    cout << dp[W] << "\n";
    return 0;
}
```
