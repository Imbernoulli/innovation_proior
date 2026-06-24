**Problem.** A relay has `n` legs (`0 .. n-1`) and exactly `n` runners. `s[i][j]` is the synergy if
runner `i` runs leg `j`. Assign each runner to one leg and each leg to one runner (a permutation),
maximizing the sum of chosen `s[i][j]`. Read `n` and the `n x n` table from stdin; print the maximum
total synergy. Bounds: `1 <= n <= 18`, `0 <= s[i][j] <= 10^9`.

**Why brute force is out.** Enumerating all `n!` assignments is correct (it is the oracle for small
`n`) but `18!` ≈ `6.4 * 10^15` is hopeless. We need to collapse equivalent partial states.

**Key idea — bitmask DP over the set of used runners.** Fill legs in index order. A partial
assignment is fully described by *which* runners are already used, because the legs they occupy are
exactly `0 .. (used-1)`. Let `dp[mask]` = best total synergy using exactly the runners in `mask`,
placed on legs `0 .. popcount(mask)-1`. The next leg to fill is `leg = popcount(mask)`. Transition:
for each free runner `i` (bit not in `mask`),

- `dp[mask | (1<<i)] = max(dp[mask | (1<<i)], dp[mask] + s[i][leg])`.

Base case `dp[0] = 0`; answer `dp[(1<<n)-1]`.

**Correctness.** Every permutation is exactly one path `0 -> ... -> full` in this DAG: at step `k`
the path appends the runner sent to leg `k`. Conversely every `n`-step path from `0` to the full
mask adds `n` distinct runners, i.e. a valid permutation. The DP maximizes `dp[mask] + s[i][leg]`
over all paths, hence over all permutations. The state is sufficient because, with legs filled in
order, the remaining problem depends only on the free runner set (and the next leg index, which the
mask determines). Iterating masks in increasing order is a valid topological order since adding a bit
strictly increases the mask, so every predecessor is finalized before its successors.

**Pitfalls.**
1. *Integer overflow (the trap).* The answer is a sum of `n` entries, up to `18 * 10^9 = 1.8 * 10^10`,
   far above `INT_MAX ≈ 2.147 * 10^9`. Using `int` for the table, the DP array, or the running sum
   silently wraps: on the sample the last addition `2*10^9 + 10^9 = 3*10^9` overflows and an `int`
   build prints `-536870912` instead of `3000000000`. Make `s`, `dp`, and the candidate `long long`.
2. *Column index.* The leg to charge a runner against is `popcount(mask)`, not the runner index and
   not a fixed column; using the wrong column quietly returns a non-optimal value.
3. *Sentinel.* Mark unreachable cells with `LLONG_MIN / 4` (not `LLONG_MIN`) and skip them, so no
   `max`/addition against the sentinel can underflow or pollute a reachable state.

**Edge cases.** `n = 1` -> output `s[0][0]`. All-zero table -> `0`. Maximum `n = 18` with all
`10^9` -> `1.8 * 10^10` (only correct in 64-bit). Ties / multiple optimal assignments -> the DP
returns the optimal value, which is all that is asked. The full mask is always reachable (complete
table), so `dp[(1<<n)-1]` is never the sentinel. `cin >>` is whitespace-agnostic and empty input is
guarded.

**Complexity.** `O(2^n * n)` time (`≈ 4.7 * 10^6` at `n = 18`) and `O(2^n)` space (`262144` cells of
`long long`). Comfortably within 2 s / 256 MB.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // empty input -> nothing to do
    vector<vector<long long>> s(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> s[i][j];

    // dp[mask] = best total synergy achievable using exactly the runners in `mask`,
    // assigned to legs 0 .. popcount(mask)-1 (legs are filled in index order).
    // We assign the runner chosen for leg = popcount(mask) next.
    const long long NEG = LLONG_MIN / 4;   // sentinel for "unreachable"
    vector<long long> dp(1 << n, NEG);
    dp[0] = 0;                              // no runners placed, no legs filled, score 0

    for (int mask = 0; mask < (1 << n); mask++) {
        if (dp[mask] == NEG) continue;      // unreachable state
        int leg = __builtin_popcount((unsigned)mask); // next leg to fill
        if (leg == n) continue;             // all legs filled
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) continue;  // runner i already used
            int nmask = mask | (1 << i);
            long long cand = dp[mask] + s[i][leg];
            if (cand > dp[nmask]) dp[nmask] = cand;
        }
    }

    cout << dp[(1 << n) - 1] << "\n";
    return 0;
}
```
