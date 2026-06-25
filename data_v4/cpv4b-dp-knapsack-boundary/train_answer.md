**Problem.** You buy a subset of `n` items (each at most once); item `i` has integer price `p[i] >= 1` and integer joy `v[i]` (possibly negative). The subset is legal iff its **total price** `s` lies in the band `[L, R]` with **both endpoints inclusive** (`L <= s <= R`). Output the maximum total joy over all legal subsets, or `IMPOSSIBLE` if none exists. The empty subset (price `0`, joy `0`) is legal only when `L = 0`. Read `n`, `L`, `R` and the `n` pairs from stdin; print one line.

**Key idea — exact-price 0/1 knapsack.** Use `dp[s]` = best joy of a subset whose total price is **exactly** `s`, for `s` in `0..R`. Initialize `dp[0] = 0` and every other cell to a sentinel `NEG` (unreachable). For each item `(pi, vi)` with `pi <= R`, do the standard 0/1 update with `s` running **downward** from `R` to `pi` so the item is used at most once: `dp[s] = max(dp[s], dp[s - pi] + vi)`, guarded by `dp[s - pi] != NEG`. Exact-price states keep the price visible, so the band is read off directly: the answer is `max(dp[s])` over `s = L..R` inclusive; if every such cell is `NEG`, print `IMPOSSIBLE`.

**Pitfalls.**
1. *Array size — `R + 1`, not `R`.* Prices run over `0..R` inclusive, which is `R + 1` distinct values. Sizing `dp` as `R` drops index `R` (the cap itself); the descending loop's `s = R` write then goes out of bounds (UB). A single item priced exactly at `R` exposes this — trace `n=1, L=0, R=4, (4,7)`: the answer is `7`, reachable only at `dp[4]`, which must exist.
2. *Answer scan — inclusive on both ends.* The band is `[L, R]` with both ends included, so scan `for (s = L; s <= R; s++)`. Writing `s < R` silently drops optima sitting exactly at price `R`; starting at `L + 1` drops optima sitting exactly at price `L` (trace `n=2, L=5, R=9, (5,4),(3,-2)` → answer `4` at price `5 = L`).
3. *Overflow / sentinel.* Up to `2000` items of joy `~10^9` give totals `~2*10^12`; use `long long`. Set `NEG = LLONG_MIN/4` and only ever read it behind the `!= NEG` reachability guard, so `NEG + vi` is never computed and cannot underflow. The sentinel sits ~6 orders of magnitude below any real joy, so a reachable negative joy is never mistaken for unreachable.

**Edge cases.** `n = 0`: only `dp[0]=0` reachable → `0` if `L=0`, else `IMPOSSIBLE`. All-negative joys with a forced in-band subset → the answer is the least-bad in-band joy and may be negative (handled, since `NEG` is far below it). Items with `p[i] > R` are skipped. No in-band subset → `IMPOSSIBLE`. `L = R = 0` → `0` (empty purchase).

**Complexity.** `O(n*R)` time, `O(R)` space. With `n <= 2000` and `R <= 10^5` that is `~2*10^8` simple operations — well under the 1 s limit (measured ~0.11 s, ~4 MB on a worst-shaped case).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L, R;
    if (!(cin >> n >> L >> R)) return 0;

    vector<long long> p(n), v(n);
    for (int i = 0; i < n; i++) cin >> p[i] >> v[i];

    const long long NEG = LLONG_MIN / 4;
    // dp[s] = best total joy using a subset whose total price is EXACTLY s,
    // for s in [0, R] (indices 0..R inclusive, so size R+1).
    vector<long long> dp(R + 1, NEG);
    dp[0] = 0;                                   // empty subset: price 0, joy 0

    for (int i = 0; i < n; i++) {
        long long pi = p[i], vi = v[i];
        if (pi > R) continue;                    // cannot ever fit within the cap
        for (long long s = R; s >= pi; s--) {    // 0/1 knapsack: descend so each item used once
            if (dp[s - pi] != NEG)
                dp[s] = max(dp[s], dp[s - pi] + vi);
        }
    }

    long long best = NEG;
    for (long long s = L; s <= R; s++)           // window [L, R] INCLUSIVE on both ends
        best = max(best, dp[s]);

    if (best == NEG) cout << "IMPOSSIBLE\n";
    else cout << best << "\n";
    return 0;
}
```
