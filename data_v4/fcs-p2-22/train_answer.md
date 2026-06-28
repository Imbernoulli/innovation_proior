**Problem.** Given `nums[0..n-1]` (`0 <= n <= 500`, `0 <= nums[i] <= 100`), burst the balloons one at a
time in any order; bursting `i` pays `nums[left] * nums[i] * nums[right]` against the neighbours still
present at that instant, with missing end-neighbours counted as a balloon painted `1`. Maximize the
total coins. Read `n` and the values from stdin, print the maximum. `n = 0` -> `0`.

**Why the obvious greedy is wrong.** Any fixed local pop-rule keyed on the painted numbers fails,
because the payout couples each balloon to whichever neighbours survive beside it — a global effect.
On `[3, 1, 5, 8]`, "burst the smallest first" collects `3*1*5 + 1*3*5 + 1*5*8 + 1*8*1 = 15+15+40+8 =
78`, but the optimal order (burst `1`, `5`, `3`, `8`) collects `3*1*5 + 3*5*8 + 1*3*8 + 1*8*1 =
15+120+24+8 = 167` — more than double. Smallest-first popped `5` with only the boundary `1` on its
left, instead of keeping `3` and `8` flanking it so `5` scores `3*5*8 = 120`. "Largest-first" is no
better. Greedy is discarded.

**Key idea — interval DP on the LAST balloon to burst.** The natural "which balloon do I burst
*first* in range `[i..j]`" recurrence is unsound: removing the first balloon makes the two
sub-ranges' neighbours depend on the order chosen in the other sub-range, so they do not decouple.
Flip it. Pad the row with virtual `1`s: `v[0] = v[n+1] = 1`, real balloons in `v[1..n]`. Define

- `dp[l][r]` = max coins from bursting every balloon strictly between padded indices `l` and `r`,
  with `l` and `r` still present as walls.

Choose `k` in `(l, r)` to be burst **last** in that open interval. At that instant every other
balloon between `l` and `r` is gone, so `k`'s neighbours are exactly the fixed walls `l` and `r`,
pinning its payout to `v[l]*v[k]*v[r]` regardless of internal order — and the two sides decouple
(walls `l,k` on the left, `k,r` on the right):

```
dp[l][r] = max over k in (l, r) of  dp[l][k] + v[l]*v[k]*v[r] + dp[k][r]
```

Empty gaps (`r = l+1`, no balloon between) have `dp = 0`. The answer is `dp[0][n+1]`.

**Two pitfalls to get right.**
1. *Last, not first.* Decide each range by its last-burst balloon so neighbours are the pinned walls
   `l, r`; the first-burst framing leaves the sub-problems' walls undefined and is wrong.
2. *Gap-length bounds with the padding.* Fill by increasing gap `len = r - l` so `dp[l][k]` and
   `dp[k][r]` (both smaller gaps) are ready first. The meaningful gaps run `len = 2 .. n+1` — *not*
   the closed-interval reflex `1 .. n`; the answer `dp[0][n+1]` has gap `n+1`, and a loop stopping at
   `len = n` leaves it `0` and prints `0` on the sample. Empty gaps `len = 1` are never assigned and
   correctly stay `0`.

**Arithmetic / complexity.** Each burst pays at most `100^3 = 10^6` and there are `n` bursts, so the
total is at most `n*10^6 = 5*10^8` for `n = 500` — it fits a 32-bit `int`, but the cube
`v[l]*v[k]*v[r]` is computed in `long long` as cheap defensive headroom. Time `O(n^3)` (`~2*10^7`
real ops at `n = 500`, about `0.02 s`); memory `O(n^2)` (`~2 MB`).

**Verification.** Differential-tested against an independent brute that simulates the bursting game by
*which balloon to burst next* (the rejected first-move framing), memoized on the surviving-position
set, over 1000+ random rows (`n <= 9`, values including `0`s and the cap) plus edge cases (`n = 0`,
single balloon, all zeros, all `100`s, the sample, alternating `100 0 ...`): 0 mismatches. Worst case
`n = 500` all-`100` runs in `~0.02 s` and yields `498010100`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> 0 coins
    vector<long long> v(n + 2);
    v[0] = 1;                              // virtual balloon "1" on the left
    v[n + 1] = 1;                          // virtual balloon "1" on the right
    for (int i = 1; i <= n; i++) cin >> v[i];

    // dp[l][r] = max coins from bursting every balloon strictly between
    // padded indices l and r (exclusive), with l and r still present.
    // Choose k in (l, r) as the LAST balloon burst in that open interval;
    // when k is burst its neighbours are exactly l and r, contributing
    // v[l] * v[k] * v[r], plus the two already-solved sub-intervals.
    vector<vector<long long>> dp(n + 2, vector<long long>(n + 2, 0));
    for (int len = 2; len <= n + 1; len++) {          // gap length r - l
        for (int l = 0; l + len <= n + 1; l++) {
            int r = l + len;
            long long best = 0;
            for (int k = l + 1; k < r; k++) {
                long long cur = dp[l][k] + v[l] * v[k] * v[r] + dp[k][r];
                if (cur > best) best = cur;
            }
            dp[l][r] = best;
        }
    }

    cout << dp[0][n + 1] << "\n";
    return 0;
}
```
