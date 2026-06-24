**Problem.** There are `m` required skills (`0 .. m-1`) and `k` contractors. Contractor `i` has fee `cost[i]` and a skill set given as the `m`-bit mask `mask[i]`. Hire a subset of contractors whose pooled (union of) skills cover all `m` skills, minimizing the total fee. Print that minimum, or `-1` if no subset covers everything. Read `m k`, then `k` lines `cost[i] mask[i]`, from stdin.

**Why the obvious greedy is wrong.** "Repeatedly hire the contractor with the best fee-per-newly-covered-skill ratio" is the textbook set-cover greedy, and it is only an *approximation*. On `m = 6` with `L = (fee 3, mask 111110b = 62)`, `H1 = (fee 2, mask 000111b = 7)`, `H2 = (fee 2, mask 111000b = 56)`: greedy first takes `L` (ratio `3/5 = 0.6`, strictly best), then is forced to add `H1` for the leftover skill `0`, total `5`; but `H1 + H2` covers `{0,1,2} ∪ {3,4,5}` for `4`. The lure `L` re-covers skills the optimal pair already supplies for free — local fee-efficiency ignores the overlap structure the union constraint cares about. Greedy is discarded.

**Key idea — bitmask DP over coverage states.** The only thing about a partial hire that constrains the future is *which skills are already covered*, so let the state be the covered-skill set `s in {0,1}^m`. Let `dp[s]` = minimum fee to reach coverage `s`. Base: `dp[0] = 0`. Transition, relaxing from a reachable `s` over every contractor:

- `dp[s | mask[i]] = min(dp[s | mask[i]], dp[s] + cost[i])`.

Answer is `dp[FULL]` with `FULL = 2^m - 1`, or `-1` if it stays infinite.

**Correctness.** Every transition goes from a set `s` to a *superset* `s | mask[i]`. As integers a superset is `>= s`, and strictly greater when it differs, so iterating `s` from `0` upward processes each state only after all relaxations into it (which come from strictly smaller states) have been applied — one forward pass suffices, like a DAG shortest path. The relaxation explores every reachable union with the cheapest fee to reach it, so `dp[FULL]` is the true optimum; if `FULL` is never reached, no subset covers all skills and the answer is `-1`.

**Pitfalls.**
1. *Sentinel overflow.* Do **not** relax out of unreachable states: `dp[s] + cost[i]` with `dp[s] = LLONG_MAX` is signed overflow (UB), and the wrapped-negative value can fabricate a "reachable" state. Guard with `if (dp[s] == INF) continue;` and set `INF = LLONG_MAX / 4` so the sentinel has slack. (A trace of `m=1, mask 0` — answer `-1` — exposes exactly this.)
2. *Processing order.* The increasing-`s` order is what makes a single pass correct; it is not arbitrary. It works precisely because transitions only increase the integer value of the state.
3. *Impossibility.* Report `-1` when `dp[FULL] >= INF`. Relaxations only lower values, so an unreached state is exactly `INF`.

**Edge cases.** A skill in nobody's mask -> `-1`. `m = 1` with a capable contractor -> its fee. Overlapping masks where two cheap pieces (`2 + 3 = 5`) beat one all-covering bundle (`10`) -> the `min` relaxation prefers the cheaper combination. Duplicate masks at different fees -> the cheaper wins automatically. Largest case `m = 18, k = 100` -> `2^18 * 100 ≈ 2.6*10^7` relaxations, well under the limit.

**Complexity.** `O(2^m * k)` time, `O(2^m)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int m, k;
    if (!(cin >> m >> k)) return 0;          // empty input -> nothing to do
    vector<long long> cost(k);
    vector<int> mask(k);
    for (int i = 0; i < k; i++) {
        cin >> cost[i] >> mask[i];           // cost, then the skill bitmask (0 .. 2^m - 1)
    }

    const int FULL = (1 << m) - 1;
    const long long INF = LLONG_MAX / 4;
    vector<long long> dp(1 << m, INF);
    dp[0] = 0;                               // covering the empty set costs nothing

    // Forward DP over coverage states: from a reachable state, try hiring each contractor.
    for (int s = 0; s <= FULL; s++) {
        if (dp[s] == INF) continue;
        for (int i = 0; i < k; i++) {
            int ns = s | mask[i];
            long long nc = dp[s] + cost[i];
            if (nc < dp[ns]) dp[ns] = nc;
        }
    }

    if (dp[FULL] >= INF) cout << -1 << "\n";
    else cout << dp[FULL] << "\n";
    return 0;
}
```
