**Problem.** A reservoir starts at level `0`. Day `i` changes the level by an integer `d[i]` (which may be positive, negative, or zero), so the level after day `i` is the prefix sum `P[i] = d[0] + ... + d[i]`, and the start level is `P[-1] = 0`. Report the **worst decline** `max over -1 <= i <= j <= n-1 of (P[i] - P[j])`. Since `i = j` is allowed, the answer is at least `0`. Read `n` and the `n` values from stdin; print the worst decline.

**Why the obvious all-pairs scan is too slow.** Computing `P[i] - P[j]` for every pair `i <= j` is obviously correct but `O(n^2)`, which is `~4*10^10` operations at `n = 2*10^5` — far over the 1-second limit. It is fine only as a reference oracle on tiny inputs.

**Key idea — prefix sum with a running peak.** For a fixed later index `j`, the drop `P[i] - P[j]` is maximized by the **highest** level `P[i]` among `i <= j`. So sweep left to right, carrying `peak = max level seen so far` (the start level `0` included) and the current level `prefix`. At each day the best decline ending there is `peak - prefix`, and the global answer is the max of that over all days, floored at `0`:

- `answer = max(answer, peak - prefix)`  (drop from the best earlier high-water mark to today)
- `peak = max(peak, prefix)`  (then fold today's level into the running peak)

Initialize `prefix = 0`, `peak = 0` (this `0` *is* `P[-1]`, the start level), and `answer = 0`. One pass, `O(n)` time, `O(1)` space.

**Correctness.** For any pair `(i, j)` with `i <= j`, `P[i] - P[j] <= peak_{<=j} - P[j]` since `peak_{<=j} >= P[i]`, and equality holds at the `i` realizing the max; taking the max over `j` therefore equals the all-pairs maximum. Seeding `peak = 0` before the loop keeps index `-1` (the start level) eligible as a peak for every day.

**Pitfalls.**
1. *Wrong base case — dropping the start level.* The starting level `0` is itself a peak. Seeding the peak from the *first day's* level (e.g. `if (i == 0) peak = prefix;`) silently drops `P[-1] = 0`. Then `[-7]` returns `0` instead of `7`, and `[-2,-3,-1,-4]` returns `8` instead of `10` — it undercounts by the start level on every all-negative or single-negative input. Seed `peak = 0` *before* the loop. Unlike maximum-subarray, an all-negative season does **not** give `0`: the level only falls, so the decline is the full fall from the start.
2. *Overflow.* With `n` up to `2*10^5` and `|d[i]|` up to `10^9`, a level — and the worst decline — reaches `~2*10^14`. Use `long long`; an `int` is a silent wrong-answer on large tests.

**Edge cases (all handled by the recurrence + `answer >= 0` floor):** `n = 0` -> `0`; a single positive day -> `0`; a single negative day `[-7]` -> `7` (start level is the peak); all non-negative or all zeros -> `0`; all negative `[-2,-3,-1,-4]` -> `10`; `n = 2*10^5` of `-10^9` -> `2*10^14`.

**Complexity.** `O(n)` time, `O(1)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;            // n = 0 (or empty input) -> answer 0

    // Level starts at 0 before any day; this initial level is a valid "peak".
    long long prefix = 0;                  // P[-1] = 0
    long long peak = 0;                    // best (max) level seen so far, including the start
    long long answer = 0;                  // i = j gives a decline of 0, so answer >= 0

    for (int i = 0; i < n; i++) {
        long long d;
        cin >> d;
        prefix += d;                       // level after day i
        // decline ending at this day = (highest earlier-or-equal level) - current level
        answer = max(answer, peak - prefix);
        peak = max(peak, prefix);          // update running peak AFTER measuring the decline
    }

    cout << answer << "\n";
    return 0;
}
```
