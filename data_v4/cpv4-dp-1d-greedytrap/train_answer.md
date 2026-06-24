**Problem.** A tightrope walker starts on platform `0` of a row of `n` platforms and must finish on
platform `n-1`. From platform `i` she may hop to `i+1` or `i+2` (never further). She pays toll
`t[i] >= 0` for every platform she lands on or starts on (so always `t[0]` and `t[n-1]`). Read `n` and
the tolls from stdin; print the minimum total toll of a legal crossing.

**Why the obvious greedy is wrong.** "From `i`, hop to whichever of `i+1`, `i+2` has the smaller toll"
fails because a hop changes which platforms are reachable next: paying for an extra landing now can
never be undone. On `t = [3, 1, 1, 9, 1, 9, 1]` greedy hops `0 -> 1 -> 2 -> 4 -> 6` and pays
`3 + 1 + 1 + 1 + 1 = 7`, but hopping `0 -> 2 -> 4 -> 6` pays only `3 + 1 + 1 + 1 = 6` by never landing
on platform `1` at all. No tie-break rule rescues greedy (e.g. `t = [2,1,100,1,1,1]` gives greedy `6`
vs optimal `5`), because knowing whether to absorb a small toll now requires knowing the tolls ahead.
Greedy is discarded.

**Key idea — linear prefix DP.** Let `dp[i]` be the minimum total toll of a legal route that *lands*
on platform `i`. Any route reaching `i` made its last hop from `i-1` (`+1`) or `i-2` (`+2`), then pays
`t[i]`:

- `dp[i] = t[i] + min(dp[i-1], dp[i-2])`   for `i >= 2`

Base cases: `dp[0] = t[0]` (the start), and `dp[1] = t[0] + t[1]` — platform `1` is reachable *only*
from `0` via a `+1` hop, so its route is forced and pays both tolls. The answer is `dp[n-1]` (she must
finish on the last platform; there is no "do nothing" option). Keeping only the last two `dp` values
gives `O(1)` memory.

**Correctness.** The two predecessors `i-1`, `i-2` are exhaustive — they are the only legal hop
lengths, so no transition is missed. Optimal substructure holds by cut-and-paste: a cheapest route to
`i` must reach its predecessor by a cheapest route (else swap in the cheaper prefix), so `min` over
the two predecessor optima is exactly `dp[i]`.

**Pitfalls.**
1. *`dp[1]` base case.* It is `t[0] + t[1]`, not `t[1]`: reaching platform `1` forces having paid for
   platform `0`. (A trace of `n = 2`, `t = [5, 7]`, returning `7` instead of `12` exposes this — the
   loop starts at `i = 2`, so the raw `dp[1]` is printed.)
2. *Tiny-`n` out-of-bounds.* The two-value carry reads `t[0]` and `t[1]`, so `n = 0` and `n = 1` must
   be guarded *before* it runs, or the code reads past the end of the array.
3. *Overflow.* With `n` up to `2*10^5` and tolls up to `10^9`, an all-`+1` route lands on every
   platform and the total reaches `~2*10^14`; use `long long`. An `int` is a silent wrong-answer.

**Edge cases.** `n = 0` -> `0`; `n = 1` -> `t[0]`; `n = 2` -> `t[0] + t[1]` (forced `0 -> 1`);
an expensive middle platform is skipped automatically (`[0, 10^9, 0] -> 0` via `0 -> 2`); all-zero
tolls -> `0`.

**Complexity.** `O(n)` time, `O(1)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // empty input -> 0 (no platforms, no tolls)
    vector<long long> t(n);
    for (auto &x : t) cin >> x;

    if (n == 0) { cout << 0 << "\n"; return 0; }
    if (n == 1) { cout << t[0] << "\n"; return 0; } // must land on the only platform

    // dp[i] = minimum total toll paid for a legal route that LANDS on platform i,
    // starting from platform 0 (always landed on) and using jumps of +1 or +2.
    // dp[0] = t[0]. To reach i you arrive from i-1 (a +1 jump) or i-2 (a +2 jump):
    //   dp[i] = t[i] + min(dp[i-1], dp[i-2]).
    // We only keep the previous two values (O(1) memory).
    long long prev2 = t[0];                 // dp[0]
    long long prev1 = t[1] + t[0];          // dp[1]: only reachable from 0 via +1
    for (int i = 2; i < n; i++) {
        long long cur = t[i] + min(prev1, prev2);
        prev2 = prev1;
        prev1 = cur;
    }

    cout << prev1 << "\n";                   // dp[n-1]: must finish on the last platform
    return 0;
}
```
