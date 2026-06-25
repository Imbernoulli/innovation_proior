**Problem.** Given an `n x n` cost matrix `c` (`0 <= n <= 20`, `0 <= c[i][j] <= 10^9`), assign each of the `n` couriers to a distinct zone (a permutation `p`), minimizing the total cost `sum_i c[i][p[i]]`. Read `n` and the matrix from stdin; print the minimum total. For `n = 0` the answer is `0`.

**Why the obvious greedy is wrong.** "Send each courier to its cheapest free zone" fails because the cheapest single edges can form an illegal assignment. On `c = [[1,2],[1,9]]` greedy sends courier 0 to zone 0 (cost 1), leaving courier 1 with zone 1 (cost 9), total `10`; but courier 0 -> zone 1, courier 1 -> zone 0 totals `3`. Local cheap-edge grabs ignore the global one-to-one constraint. Greedy is discarded.

**Key idea — bitmask DP over used zones.** Place couriers in the fixed order `0, 1, 2, ...`. The number of couriers already placed equals the number of used zones equals `popcount(mask)`, so the mask alone determines which courier is next. Define `dp[mask]` = minimum cost to assign couriers `0 .. popcount(mask)-1` onto exactly the zones in `mask`. Transition, placing courier `k = popcount(mask)` into any unused zone `z`:

- `dp[mask | (1<<z)] = min( dp[mask | (1<<z)], dp[mask] + c[popcount(mask)][z] )`.

Base `dp[0] = 0`; answer `dp[(1<<n) - 1]`. This is `O(2^n * n)` time and `O(2^n)` memory; for `n = 20` that is about `2.1*10^7` transitions and `8` MB — comfortably inside a 2 s limit.

**Pitfalls to get right.**
1. *32-bit overflow (the trap this problem is built on).* With `n` up to `20` and entries up to `10^9`, the optimal total can reach `2*10^10`, nearly ten times `INT_MAX = 2147483647`. The overflow is invisible on small cases and only appears once a partial sum first crosses `2.147*10^9`. Concretely, with `n = 3` and every entry `8*10^8`: the accumulator reads `8*10^8`, then `1.6*10^9` (both fit), then on the third addition `2.4*10^9` wraps in 32-bit two's complement to `-1894967296` — a negative cost, with no crash. Hold the matrix, the `dp` array, and every intermediate sum in `long long`.
2. *INF sentinel.* `INF` must exceed every reachable total (`> 2*10^10`), so `1e9` is far too small and would clobber real states; use something like `4e18`. Guard `if (dp[mask] == INF) continue;` before reading `dp[mask]`, so INF is never an operand of `+` and cannot itself overflow.
3. *Next-courier index.* When `popcount(mask) == n` (the full mask) skip the body; otherwise `c[courier][...]` would be indexed with `courier == n`. The guard also makes `n = 0` (only `mask = 0`, `courier = 0 = n`) return `0` cleanly.

**Edge cases.** `n = 0` -> `0` (the `courier == n` guard skips everything, output `dp[0] = 0`); `n = 1` -> `c[0][0]`; all-equal matrices -> `n * value` via the `min` over ties; large `n = 20` with entries near `10^9` -> total about `2*10^10`, exact in `long long`.

**Complexity.** `O(2^n * n)` time, `O(2^n)` memory.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<vector<long long>> c(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> c[i][j];

    const long long INF = (long long)4e18;
    // dp[mask] = minimum cost to have assigned couriers 0..popcount(mask)-1
    // to exactly the set of zones in `mask`.
    int full = (1 << n);
    vector<long long> dp(full, INF);
    dp[0] = 0;
    for (int mask = 0; mask < full; mask++) {
        if (dp[mask] == INF) continue;
        int courier = __builtin_popcount(mask); // next courier to place
        if (courier == n) continue;
        for (int z = 0; z < n; z++) {
            if (mask & (1 << z)) continue;       // zone z already used
            int nmask = mask | (1 << z);
            long long cand = dp[mask] + c[courier][z];
            if (cand < dp[nmask]) dp[nmask] = cand;
        }
    }

    cout << dp[full - 1] << "\n";
    return 0;
}
```
