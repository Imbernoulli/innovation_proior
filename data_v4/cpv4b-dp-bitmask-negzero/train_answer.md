**Problem.** A festival has `n` parallel stages and `n` candidate artists; booking artist `i` on
stage `j` yields a net reward `p[i][j]` that may be negative or zero. Each artist performs at most
once, each stage hosts at most one artist, and no stage or artist must be used. Choose a *partial*
matching (subset of artists, each on a distinct stage) maximizing total reward. The empty booking is
allowed, so the answer is at least `0`. Read `n` and the `n x n` matrix from stdin; print the maximum.

**Key idea — stage sweep with a subset DP over artists.** With `n <= 18`, index the state by the set
of artists already booked. Sweep stages `s = 0..n-1`. Let `dp[mask]` = best total reward after
deciding stages `0..s-1`, where `mask` is the set of booked artists. At stage `s`, from a reachable
`dp[mask]`:

- *Leave stage `s` empty:* `ndp[mask] = max(ndp[mask], dp[mask])`  (reward `+0`, no artist consumed).
- *Book a free artist `a`:* `ndp[mask | (1<<a)] = max(..., dp[mask] + p[a][s])` for each clear bit `a`.

Initialize `dp[0] = 0` (before any stage, nobody booked) and all other masks to `-inf`. Answer:
`max over all reachable masks of dp[mask]`, floored at `0`.

**Pitfalls.**
1. *Missing the "leave empty" branch (a base/skip bug).* Without `ndp[mask] = max(ndp[mask],
   dp[mask])`, the reward-`0` empty state `dp[0]` is destroyed after the first stage, silently
   re-imposing *full* assignment: every later booking must be padded with earlier losses. A trace of
   `n=2`, `p=[[-1,5],[-1,-1]]` returns `4` instead of the correct `5` (skip stage 0, book artist 0 on
   stage 1 for `+5`). The skip branch is what makes "partial" first-class.
2. *Wrong aggregation / sign on the empty corner.* Reading the answer from the full mask
   (`ans = dp[full-1]`) reports the best *permutation*, which on an all-loss matrix is negative — e.g.
   `p=[[-3,-1],[-4,-2]]` gives `dp[full-1] = -5`, but booking nobody nets `0`. Maximize over *all*
   masks and floor at `0`. `dp[0] = 0` always survives, encoding the empty booking.
3. *Overflow.* With `n = 18` and `|p| <= 10^9`, a full booking reaches `~1.8 * 10^10`, past 32-bit;
   use `long long`. Guard the `-inf` sentinel so `p[a][s]` is never added to it.

**Edge cases.** `n = 0` -> `0`; a single negative -> `0`; all-negative matrix -> `0` (book nobody);
zero entries are neutral (neither force nor forbid a booking). All are handled by `dp[0]=0` plus the
`0`-floored max.

**Complexity.** `O(n^2 * 2^n)` time (`~8.5 * 10^7` updates at `n=18`), `O(2^n)` memory (two arrays).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;            // n = 0 -> empty roster -> 0
    vector<vector<long long>> p(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) cin >> p[i][j];

    // We sweep stages s = 0..n-1. dp[mask] = best total reward achievable after deciding
    // stages 0..s-1, where `mask` is the set of artists already booked. Each stage may be
    // left empty (booking nobody, +0) or given one still-free artist (+p[artist][stage]).
    // Partial rosters are allowed, so the empty roster (dp all skipped) keeps the value 0.
    const long long NEG = LLONG_MIN / 4;
    int full = 1 << n;
    vector<long long> dp(full, NEG);
    dp[0] = 0;                            // before any stage: nobody booked, reward 0

    for (int s = 0; s < n; s++) {
        vector<long long> ndp(full, NEG);
        for (int mask = 0; mask < full; mask++) {
            if (dp[mask] == NEG) continue;
            // Option 1: leave stage s empty.
            if (dp[mask] > ndp[mask]) ndp[mask] = dp[mask];
            // Option 2: book one free artist a on stage s.
            for (int a = 0; a < n; a++) {
                if (mask & (1 << a)) continue;
                int nm = mask | (1 << a);
                long long cand = dp[mask] + p[a][s];
                if (cand > ndp[nm]) ndp[nm] = cand;
            }
        }
        dp.swap(ndp);
    }

    long long ans = 0;                    // empty roster is always allowed
    for (int mask = 0; mask < full; mask++)
        if (dp[mask] != NEG && dp[mask] > ans) ans = dp[mask];

    cout << ans << "\n";
    return 0;
}
```
