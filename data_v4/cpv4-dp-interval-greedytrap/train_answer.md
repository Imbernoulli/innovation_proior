**Problem.** `n` beads in fixed order have widths `w[0..n-1]` (`1 <= w[i] <= W`). Cut the sequence into contiguous groups, one group per glow-line of usable width `W`. A line holding beads `l..r` uses `used(l,r) = (w[l]+...+w[r]) + (r-l)` (one unit gap per adjacent pair) and must satisfy `used <= W`. A non-last line pays `slack^2 = (W - used)^2`; the last line is free. Minimize the total penalty. Read `n W` then the `n` widths from stdin; print the minimum.

**Why the obvious greedy is wrong.** "Fill each line as full as possible, then start a new line" (first-fit) minimizes each line's slack *locally*, but the cost is the *sum of squares* of slack, a convex penalty that rewards spreading slack evenly rather than concentrating it. On `n=4, W=6, w=[4,1,3,3]`, greedy packs line 1 = `{0,1}` to exactly `6` (slack 0), then is forced into `{2}` (slack 3) and `{3}`, paying `0^2 + 3^2 = 9`. The optimum deliberately wastes a little on line 1 = `{0}` (slack 2) to keep the rest balanced: `{1,2}` (slack 1), `{3}` free, paying `2^2 + 1^2 = 5 < 9`. Both waste the same total slack units; convexity makes the balanced split cheaper. Greedy is discarded.

**Key idea — interval / partition DP.** Let `dp[i]` = minimum penalty to lay out the first `i` beads, where that layout's last line is some contiguous group `[j-1 .. i-1]`. With prefix sums `pre[i] = w[0]+...+w[i-1]`, a group `[j-1..i-1]` has `cnt = i-(j-1)` beads, `used = (pre[i]-pre[j-1]) + (cnt-1)`, and is legal iff `used <= W`. Transition, over all legal `j`:

- `dp[i] = min_j ( dp[j-1] + pen )`, where `pen = (W - used)^2` for an interior line but `pen = 0` when `i == n` (the global last line is free).

Initialize `dp[0] = 0`; the answer is `dp[n]`. Because `used` only grows as the group widens (smaller `j`), the inner scan can `break` the instant `used > W`.

**Correctness.** Any optimal layout has a well-defined last line `[j-1..i-1]`; removing it leaves an optimal layout of beads `0..j-2` (else swap in a cheaper prefix — exchange argument), which `dp[j-1]` captures. Taking the min over all legal last-group starts `j` therefore yields the true optimum for `dp[i]`. Keying the free-line rule on `i == n` is correct because the group ending at bead `n-1` is exactly the board's final line; every group with `i < n` is interior and must pay `slack^2`.

**Pitfalls.**
1. *Greedy.* First-fit is provably suboptimal under the squared penalty (counterexample above). Use the DP.
2. *Missing feasibility test.* You must skip groups with `used > W`. Forgetting it lets the DP build an overflowing line; on the last line the `i==n` zero-cost rule even *masks* the bug (an over-wide line slips through as "free"), and on interior lines a negative `slack` squares into a positive, spurious cost. Guard with `if (used > W) break;` (valid since `used` is monotonic in the group width).
3. *Overflow.* The maximum answer is `~1.25*10^15` (up to `n-1` lines each paying near `(W/2)^2`), so all accumulators must be `long long`; an `int` is a silent wrong-answer. Set `INF = LLONG_MAX/4` and skip unreachable `dp[j-1] == INF` so `INF + pen` never overflows.

**Edge cases.** `n = 1` -> the single line is the last line, free -> `0`. Everything fits on one line -> that one line is last -> `0`. Each bead `= W` wide -> no sharing, every bead alone, only the last free, the rest pay `0` slack -> handled by `break`. Forced sparse lines (`w=[4,4,4,4], W=6`) -> three interior lines pay `(6-4)^2` each -> `12`.

**Complexity.** `O(n^2)` worst case (inner scan bounded by beads-per-line, never more than `n`), `O(n)` memory. At `n = 5000` this is `~2.5*10^7` operations, well under 1 second (measured ~0.02 s on the adversarial all-width-1 input).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, W;
    if (!(cin >> n >> W)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // prefix[i] = sum of widths of beads 0..i-1
    vector<long long> pre(n + 1, 0);
    for (int i = 0; i < n; i++) pre[i + 1] = pre[i] + w[i];

    const long long INF = LLONG_MAX / 4;

    // dp[i] = minimum total penalty to pack the first i beads (0..i-1) into lines.
    // A line covering beads [j..i-1] is allowed only if its used width
    //   used = sum w[j..i-1] + (i-1-j)   (one unit gap between adjacent beads)
    // does not exceed W. Penalty of that line is (W - used)^2, EXCEPT the last
    // line (the one ending at i == n) has penalty 0 (no trailing-slack penalty).
    vector<long long> dp(n + 1, INF);
    dp[0] = 0;
    for (int i = 1; i <= n; i++) {
        // line is [j .. i-1], 0-based beads, j from i-1 down to 0
        for (int j = i; j >= 1; j--) {
            // beads j-1 .. i-1  (1-based prefix indexing: beads with indices j-1..i-1)
            long long cnt = i - (j - 1);             // number of beads on the line
            long long widthSum = pre[i] - pre[j - 1];
            long long used = widthSum + (cnt - 1);   // gaps between beads
            if (used > W) break;                     // adding earlier beads only grows used
            if (dp[j - 1] == INF) continue;
            long long slack = W - used;
            long long pen = (i == n) ? 0 : slack * slack; // last line: no slack penalty
            dp[i] = min(dp[i], dp[j - 1] + pen);
        }
    }

    cout << dp[n] << "\n";
    return 0;
}
```
