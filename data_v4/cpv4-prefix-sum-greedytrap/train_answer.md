**Problem.** Given `n` daily profits `a[0..n-1]` (values may be negative) and a minimum lot length `L`, choose non-overlapping contiguous lots, each spanning at least `L` days, to maximize the sum of their values; harvesting nothing is allowed, so the answer is at least `0`. Read `n`, `L`, and the values from stdin; print the maximum total. (`L` may exceed `n`, in which case no lot fits and the answer is `0`.)

**Why the obvious greedy is wrong.** "Repeatedly take the single maximum-sum lot of length `>= L` over the still-free days, then recurse on the free runs" feels optimal but fails because lot selection is a global packing decision. On `L = 2`, `a = [9, -2, -5, 8, -5, 9]`, greedy's single best block is the whole array `[0,5] = 14`, which then consumes everything. But declining to harvest the interior losses and taking two lots `[0,1] = 9-2 = 7` and `[3,5] = 8-5+9 = 12` gives `19 > 14`: a globally-best block can straddle a valley the optimum prefers to leave uncovered. Greedy is discarded.

**Key idea — prefix-sum boundary DP.** Precompute `P[i] = a[0] + ... + a[i-1]`, so any lot `[j, i-1]` is worth `P[i] - P[j]` in `O(1)`. Let `dp[i]` be the best total over the first `i` days. At boundary `i`, day `i-1` is either left uncovered or it ends a lot `[j, i-1]` with length `i - j >= L` (so `j <= i - L`):

- `dp[i] = max( dp[i-1],  P[i] + max_{0 <= j <= i-L} ( dp[j] - P[j] ) )`.

Because `dp[j] - P[j]` depends only on `j` and the eligible set grows by exactly one (`j = i - L`) as `i` advances, carry a running `best = max of (dp[j] - P[j])` and extend it each step. That makes the inner `max` `O(1)` and the whole algorithm `O(n)`. Answer: `dp[n]`, which is `>= 0` because `dp[0] = 0` and "leave uncovered" is always available.

**Correctness.** Every valid selection has a last lot ending at some boundary `i`, or no lot at all; the recurrence enumerates exactly these by either skipping day `i-1` or closing a length-`>= L` lot there, with the prefix `[0, j)` handled optimally by `dp[j]`. The running `best` ranges over precisely the `j` with `j <= i - L` (each folded in at `i = j + L`), so no shorter-than-`L` lot is ever formed.

**Pitfalls.**
1. *Greedy.* The max-block greedy is wrong (counterexample above); use the DP.
2. *Eligibility offset.* Fold `j` into `best` at boundary `i = j + L`, i.e. `j = i - L` with no `+1`. Using `i - L + 1` admits illegal length-1 lots (trace `n=2, L=2, a=[3,4]`: the wrong offset harvests a single day at `i=1`).
3. *Sentinel.* Guard the lot-closing branch with `best > NEG/2` so you never form `P[i] + sentinel` when no `j` is eligible (e.g. `L > n`); correctness then does not lean on the sentinel being "negative enough."
4. *Overflow.* With `n` up to `2*10^5` and `|a[i]|` up to `10^9`, totals reach `~2*10^14`; use `long long` for `P`, `dp`, and `best`. An `int` is a silent wrong answer on large tests.

**Edge cases (all handled by the recurrence + the guard):** `n = 0` -> `0`; `L > n` (no lot fits) -> `0`; `L = n` (whole array or nothing); `L = 1` (sum of all positive runs); all-negative days -> `0`.

**Complexity.** `O(n)` time, `O(n)` space (`O(1)` extra beyond the prefix and `dp` arrays).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Prefix sums: P[i] = a[0] + ... + a[i-1], so sum(a[j..i-1]) = P[i] - P[j].
    vector<long long> P(n + 1, 0);
    for (int i = 0; i < n; i++) P[i + 1] = P[i] + a[i];

    // dp[i] = best total over the first i days (a[0..i-1]) choosing non-overlapping
    // intervals each of length >= L; we always allow choosing nothing, so dp[i] >= 0.
    // Transition at day boundary i:
    //   - leave day i-1 uncovered: dp[i] = dp[i-1]
    //   - end an interval [j, i-1] of length (i - j) >= L: dp[i] = dp[j] + (P[i] - P[j])
    //       = (P[i]) + max over valid j of (dp[j] - P[j]).
    // Maintain best = max over j (0 <= j <= i - L) of (dp[j] - P[j]) incrementally.
    const long long NEG = LLONG_MIN / 4;
    vector<long long> dp(n + 1, 0);
    long long best = NEG; // best of (dp[j] - P[j]) for j allowed at current i
    for (int i = 1; i <= n; i++) {
        // A new candidate j = i - L becomes available the moment i reaches that j + L.
        int j = i - (int)L;
        if (j >= 0) best = max(best, dp[j] - P[j]);
        dp[i] = dp[i - 1];                       // skip day i-1
        if (best > NEG / 2) dp[i] = max(dp[i], P[i] + best); // close an interval here
    }

    cout << dp[n] << "\n";
    return 0;
}
```
