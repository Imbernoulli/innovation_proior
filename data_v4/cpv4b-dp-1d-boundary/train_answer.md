**Problem.** A frog crosses `n` stepping stones `0..n-1`, starting on `0` and finishing on `n-1`. From stone `i` it leaps forward to `j` with `1 <= j - i <= D` (reach `D` is **inclusive**). Standing on stone `j` costs toll `c[j]` (the start `0` pays its toll too); a cracked stone has `c[j] = -1` and may never be landed on. Read `n`, `D`, and the tolls from stdin; print the minimum total toll to stand on `n-1`, or `-1` if it is unreachable.

**Key idea — windowed 1-D DP with a sliding-window minimum.** Let `dp[j]` be the least toll to be standing on stone `j`. The legal predecessors of a landable stone `j` are exactly the indices in the inclusive window

- `dp[j] = c[j] + min(dp[i])` over `i in [j - D, j - 1]`, taken only if `c[j] >= 0` and some `dp[i]` is finite.

Base case: `dp[0] = c[0]` if `c[0] >= 0`, else `INF` (cracked start ⇒ no crossing). A literal scan of the window is `O(n*D)` and times out, so maintain a monotonic deque of predecessor indices (increasing index, increasing `dp`); its front is `argmin dp[i]` over the current window, giving `O(n)`. Answer is `dp[n-1]`, or `-1` if `INF`.

**Pitfalls.**
1. *Inclusive lower boundary (the decisive one).* The window is `[j-D, j-1]` with `j-D` **inclusive** — a gap of exactly `D` is a legal full-reach leap. The deque front-eviction must drop only indices with `dq.front() < j - D`; writing `<= j - D` discards the gap-`==D` predecessor and wrongly reports full-reach crossings impossible. A trace of `n=2, D=1, c=[0,5]` returning `-1` instead of `5` exposes exactly this `<=`-vs-`<` slip.
2. *Cracked stones and feasibility.* Only push `j` into the deque when `dp[j] < INF` (landable and reachable); never land on `c[j] = -1`; guard the addition with `dp[front] < INF` so `INF` is never propagated. Handle a cracked start (`c[0] = -1`) and a cracked end (`c[n-1] = -1`) — both give `-1`.
3. *Overflow.* `n` up to `2*10^5` and tolls up to `10^9` make totals reach `~2*10^14`; use `long long` for tolls, `dp`, and `D`. An `int` is a silent wrong-answer on large tests.

**Edge cases.** `n = 1` (start is the end ⇒ `c[0]`, or `-1` if cracked); `D = 1` (must step on every stone ⇒ sum of tolls, or `-1` if any interior stone is cracked); cracked start/end ⇒ `-1`; a cracked stone reachable only by a gap-`==D` leap (handled by the inclusive boundary).

**Complexity.** `O(n)` time (each index enters and leaves the deque once), `O(n)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long D;
    if (!(cin >> n >> D)) return 0;
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    const long long INF = LLONG_MAX / 4;

    // dp[j] = minimum total toll to be standing on stone j, having started on
    // stone 0 (whose toll is paid). A leap from i to j is legal iff stone j is
    // not broken and 1 <= j - i <= D (the reach D is INCLUSIVE: landing exactly
    // D ahead is allowed). A toll c[j] < 0 marks a broken stone (cannot land).
    vector<long long> dp(n, INF);

    // Stone 0 is the start. If it is broken the frog cannot even begin.
    if (c[0] >= 0) dp[0] = c[0];

    // Sliding-window minimum over the legal predecessor range [j-D, j-1].
    // The deque holds indices i with increasing index and increasing dp[i].
    deque<int> dq;
    for (int j = 0; j < n; j++) {
        // Drop predecessors that are now out of reach: i < j - D means the gap
        // j - i > D, which is illegal. The boundary i == j - D stays (gap == D).
        while (!dq.empty() && dq.front() < j - D) dq.pop_front();

        if (j > 0 && c[j] >= 0 && !dq.empty() && dp[dq.front()] < INF)
            dp[j] = dp[dq.front()] + c[j];

        // Push j as a future predecessor only if we can actually stand on it.
        if (dp[j] < INF) {
            while (!dq.empty() && dp[dq.back()] >= dp[j]) dq.pop_back();
            dq.push_back(j);
        }
    }

    if (dp[n - 1] >= INF) cout << -1 << "\n";
    else cout << dp[n - 1] << "\n";
    return 0;
}
```
