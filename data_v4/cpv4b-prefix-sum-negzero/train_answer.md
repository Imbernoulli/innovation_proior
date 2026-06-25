**Problem.** A reservoir gauge reads `0` before the season, then changes by signed integers
`d[0..n-1]` each day, so the level after day `k` is the prefix sum `L[k] = d[0]+...+d[k-1]` with
`L[0] = 0`. Report the maximum drawdown `max over 0 <= i <= j <= n of (L[i] - L[j])` — the deepest
fall of the level from any earlier reading (including the pre-season `0`) to any later one. Since
`i = j` is allowed, the answer is `>= 0`. Read `n` and the deltas from stdin, print the maximum
drawdown.

**Key idea — single pass with a running peak.** For a fixed later day `j`, the deepest fall ending
there uses the *largest* earlier-or-equal reading, the running peak `peak_j = max(L[0..j])`. So the
best drawdown ending at `j` is `peak_j - L[j]`, and

```
maxDrawdown = max over j of ( peak_j - L[j] ).
```

(Proof both ways: each `peak_j - L[j]` is a real pair `i* <= j`; and any pair has `L[i] <= peak_j`, so
`L[i]-L[j] <= peak_j-L[j]`.) Walk left to right carrying `level` (the prefix sum) and `peak`; at each
day measure the fall `peak - level`, then update `peak`. `O(n)` time, `O(1)` space.

**Pitfalls.**
1. *Sign of the drop.* A drawdown is `earlier_high - later_low = peak - level`, not `level - min`. The
   `level - min` version computes the maximum *rise* (the maximum-subarray / max-profit mirror) and is
   invisible on all-rising inputs; a trace of `[10, -30]` returns `10` instead of the correct `30`.
2. *Base case of the peak.* Seed `peak = L[0] = 0`, the pre-season reading — not `-infinity` and not
   `d[0]`. The legitimate fall measured *from the reference* is what makes a single outflow day answer
   its magnitude and an all-negative log answer its full accumulated decline; a `-infinity` seed loses
   both.
3. *Overflow.* With `n` up to `2*10^5` and `|d[i]|` up to `10^9`, the level reaches `~2*10^14` and a
   drawdown `~4*10^14`; use `long long`. An `int` is a silent wrong-answer on large tests.

**Edge cases (all handled by seeding `best = 0`, `peak = 0` and letting `peak - level` go negative
harmlessly):** `n = 0` -> `0`; a log that only rises -> `0`; a single outflow day `[-5]` -> `5`; all
negatives `[-3,-1,-4]` -> `8` (the whole decline); all zeros -> `0`.

**Complexity.** `O(n)` time, `O(1)` extra space; one pass, reads deltas on the fly.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;            // n = 0 (or empty input) -> no day, answer 0

    // L is the running level (a prefix sum of the daily deltas), starting at L[0] = 0
    // BEFORE any day. peak is the maximum level seen so far, including day 0.
    // The maximum drawdown is max over j of (peak_up_to_j - L[j]); i = j gives 0,
    // so the answer is never negative even if the level only ever rises.
    long long level = 0;                  // L[0] = 0, the reference before day 1
    long long peak = 0;                   // best level seen so far == L[0]
    long long best = 0;                   // i = j allowed -> drawdown >= 0

    for (int i = 0; i < n; i++) {
        long long d;
        cin >> d;
        level += d;                       // L[i+1] = L[i] + d[i]
        if (peak - level > best) best = peak - level;   // drop from the running peak
        if (level > peak) peak = level;   // update peak AFTER measuring the drop
    }

    cout << best << "\n";
    return 0;
}
```
