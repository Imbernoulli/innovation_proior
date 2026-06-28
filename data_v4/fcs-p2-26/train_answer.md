**Problem.** Given `a[0..n-1]` (values may be negative), pick a subsequence — positions kept in their original left-to-right order — whose values are **strictly increasing**, maximizing the sum of the chosen values; the empty subsequence is allowed, so the answer is at least `0`. Read `n` and the values from stdin, print the maximum sum.

**Why this is not the LIS problem.** It looks like longest-increasing-subsequence, but the objective is the *sum*, not the *length*. The longest chain need not be the heaviest: on `[1, 100, 2, 3, 4, 5, 6]` the answer is `101` (the chain `1, 100`), which beats the longest chain `1, 2, 3, 4, 5, 6` (sum `21`). Any instinct borrowed from LIS has to be re-justified for sum.

**Why the obvious greedy is wrong.** The tempting rule — "scan left to right, keep extending the chain by any element larger than the last one taken" — fails because the decision is local while both the ordering constraint and the sum objective are global. On `[10, 1, 2, 3, 4, 5]` greedy grabs `10` first, which raises the chain's last value to `10` and locks out every later element, finishing at `10`; but the chain `1, 2, 3, 4, 5` sums to `15`. Negatives break it further: on `[-5, -4, -3]` greedy chains all three to `-12`, worse than the empty subsequence `0`. Greedy is discarded.

**Key idea — quadratic per-ending-position DP.** For each position `i`, let `dp[i]` be the maximum sum of a strictly increasing subsequence that ends *exactly* at `i`. Such a subsequence is either just `a[i]` (a fresh start) or some chain ending at an earlier `j < i` with `a[j] < a[i]`, extended by `a[i]`:

- `dp[i] = a[i] + max(0, max over j<i with a[j] < a[i] of dp[j])`

The inner `max(0, ...)` is the "start a fresh chain at `i`, take no predecessor" option — without it, all-negative arrays would be forced to chain negatives and only get worse. The answer is `max(0, max_i dp[i])`, the outer `0` being the empty subsequence.

**Two pitfalls to get right.**
1. *Strict comparison.* The predecessor test must be `a[j] < a[i]`, not `<=`. The chain is *strictly* increasing, so equal values may not chain. (A trace of `[2, 2]` returning the illegal `4` exposes a non-strict `<=`; the correct answer is `2`.)
2. *Overflow.* With `n` up to `5000` and `|a[i]|` up to `10^9`, a chain's sum can reach `~5*10^12`; use `long long`. An `int` is a silent wrong-answer on large tests.

**Edge cases (all handled by the recurrence + the two `0` seeds):** `n = 0` -> `0`; a single negative -> `0`; all negatives -> `0`; all-equal -> the single value (no chain of length 2); strictly decreasing -> the single largest value.

**Complexity.** `O(n^2)` time, `O(n)` space. At `n = 5000` that is `25*10^6` operations — about ten milliseconds — so the simple provable DP is also comfortably fast; no `O(n log n)` weighted-LIS structure is needed at this scale, and reaching for one would only reintroduce bug risk for no benefit.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // dp[i] = max sum of a STRICTLY increasing subsequence that ends exactly at i.
    // A subsequence ending at i either starts at i (sum a[i]) or extends some j<i
    // with a[j] < a[i] (sum dp[j] + a[i]); we keep the best such predecessor.
    long long answer = 0;                  // empty subsequence is always allowed
    vector<long long> dp(n);
    for (int i = 0; i < n; i++) {
        long long best = 0;                // 0 = "no predecessor", start fresh at i
        for (int j = 0; j < i; j++) {
            if (a[j] < a[i] && dp[j] > best) best = dp[j];
        }
        dp[i] = best + a[i];
        if (dp[i] > answer) answer = dp[i];
    }

    cout << answer << "\n";
    return 0;
}
```
