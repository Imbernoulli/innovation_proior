**Problem.** There are `m` channels (`m <= 18`) and `k` bursts (`k <= 200`). Firing burst `j` costs `c[j]` and delivers the channel set `mask[j]`; redundant re-delivery is harmless. Choose a subset of bursts whose masks cover all `m` channels at minimum total cost, or print `-1`. Read `m`, `k`, then each burst's cost, channel count, and channel indices from stdin; print the minimum cost (`0` when `m = 0`).

**Why the obvious greedy is wrong.** "Repeatedly fire the burst with the best cost-per-newly-covered-channel ratio" is the classic set-cover heuristic — and it is only an *approximation*, not exact. On `m = 4` with bursts `{0,1,2}`@5, `{0,1}`@4, `{2,3}`@4: greedy grabs `{0,1,2}` first (ratio `5/3 ≈ 1.67`, the best), then must pay `4` to clean up the stranded channel `3`, total `9`. The optimum partitions cleanly: `{0,1}`@4 + `{2,3}`@4 = `8`. The high-coverage burst looked efficient but stranded an expensive leftover. Greedy is discarded.

**Key idea — bitmask DP over coverage states.** Since `m <= 18`, a set of delivered channels is one of `2^m <= 262144` subsets, each an `int`. Let `dp[S]` be the minimum cost to reach coverage `S`, with `dp[0] = 0` and all else `+inf`. Relax forward: from any reachable `S`, firing burst `j` reaches `S | mask[j]` at extra cost `c[j]`:

- `dp[S | mask[j]] = min(dp[S | mask[j]], dp[S] + c[j])`.

Answer is `dp[(1<<m)-1]`, or `-1` if it stayed infinite. Complexity `O(2^m * k) ≈ 5.2*10^7`, about `0.1 s` at the limit.

**Pitfalls to get right.**
1. *Iteration order.* `S | mask[j] >= S` always, because OR only sets bits. So every useful transition goes to a numerically larger state, and you must process `S` in **ascending** order — then `dp[S]` is final when you use it. A descending pass (a reflex from subset-sum DPs) consumes states before they are relaxed and prints `-1` on solvable inputs.
2. *Sentinel overflow.* Use `INF = LLONG_MAX/4`, not `LLONG_MAX`. With the cleaner sentinel, even an accidental `INF + cost` stays positive and never wins a `min`. Use `long long` throughout; totals can exceed 32-bit.
3. *No-op transitions.* Skip `S | mask[j] == S` (a burst whose mask is already covered, including a zero-channel burst) — it only adds cost.

**Edge cases.** `m = 0` -> `full = 0`, `dp[0] = 0`, answer `0` (vacuously covered). `k = 0` with `m > 0` -> `dp[full]` unreachable -> `-1`. Zero-channel bursts (`t = 0`) are ignored by the `ns == S` skip. Heavy overlap: the DP naturally takes the cheaper redundant burst. `2^18` `long long`s is ~2 MB, well inside 256 MB.

**Complexity.** `O(2^m * k)` time, `O(2^m)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m, k;
    if (!(cin >> m >> k)) return 0;

    vector<int> mask(k, 0);
    vector<long long> cost(k, 0);
    for (int j = 0; j < k; j++) {
        long long c;
        int t;
        cin >> c >> t;
        cost[j] = c;
        int mk = 0;
        for (int s = 0; s < t; s++) {
            int ch;
            cin >> ch;
            mk |= (1 << ch);
        }
        mask[j] = mk;
    }

    const long long INF = LLONG_MAX / 4;
    int full = (1 << m) - 1;
    vector<long long> dp(1 << m, INF);
    dp[0] = 0;

    // dp[S] = minimum total cost of a set of bursts whose union of delivered
    // channels is exactly S' >= S, reached by accumulating bursts. We process
    // states in increasing order; from state S we may fire any burst j, moving
    // to S | mask[j] at additional cost cost[j]. Firing the same burst twice is
    // never beneficial, so monotone forward relaxation suffices.
    for (int S = 0; S <= full; S++) {
        if (dp[S] == INF) continue;
        long long base = dp[S];
        for (int j = 0; j < k; j++) {
            int ns = S | mask[j];
            if (ns == S) continue;            // adds nothing new
            long long nc = base + cost[j];
            if (nc < dp[ns]) dp[ns] = nc;
        }
    }

    if (dp[full] >= INF) cout << -1 << "\n";
    else cout << dp[full] << "\n";
    return 0;
}
```
