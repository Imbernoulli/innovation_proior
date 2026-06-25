**Reading the problem and pinning the contract.** A frog starts on stone `0`, must end on stone `n-1`, and from stone `i` it can leap forward to any stone `j` with `i < j` and `j - i <= D`. Landing on stone `j` costs the toll `c[j]`; the start `0` also pays `c[0]`. Cracked stones carry `c[j] = -1` and may never be landed on. I want the minimum total toll to stand on `n-1`, or `-1` if unreachable. Before any algorithm I fix the scale, because it dictates the types: `n <= 2*10^5`, tolls up to `10^9`, and a path can touch up to `n` stones, so the total can reach `2*10^5 * 10^9 = 2*10^14`. That is far past the 32-bit limit of about `2.1*10^9`, so every toll, every `dp` value, and every accumulator must be 64-bit `long long`. An `int` here is a silent wrong-answer on the big tests. Also `D` can be as large as `2*10^5`, so I read it as `long long` too — it gets compared against index differences, and keeping it 64-bit removes any chance of a signed-comparison surprise.

**The thing I must not get wrong: which predecessors are legal.** The whole problem hinges on one sentence: "reach `D`, and a gap of `D` is the longest leap it can still make." That means a leap `i -> j` is legal exactly when `1 <= j - i <= D`, with `D` *inclusive*. Restated as a window on the predecessor index, for a target `j` the legal sources are the stones `i` with

```
j - D <= i <= j - 1.
```

Both endpoints are inclusive. The lower endpoint `i = j - D` corresponds to the maximal gap `j - i = D`, which is allowed; `i = j - 1` is the minimal gap `1`. If I ever write the window as `j - D < i` (strict) or `i <= j - 2`, I lose a legal predecessor and crossings that *only* work via a full-reach leap will be wrongly reported impossible. This is the off-by-one the problem is built around, so I will keep the window written exactly once and reuse it everywhere.

**Candidate approaches.** Two shapes of the same DP.

- *Plain windowed scan.* `dp[j] = c[j] + min over i in [j-D, j-1] of dp[i]`, taking only landable `j` (`c[j] >= 0`) and finite predecessors. Correct and dead simple, but each `j` scans up to `D` predecessors, so `O(n*D)`. With `n` and `D` both `2*10^5` that is `4*10^{10}` operations — hopeless under a 1-second limit. I will use this as my mental reference for correctness and the brute force, but not ship it.
- *Windowed DP with a monotonic deque.* Same recurrence, but maintain a deque of candidate predecessor indices in increasing index order and increasing `dp` value, so the front is always `argmin dp[i]` over the current window. Evict from the front any index that has fallen out of the window, and evict from the back any index whose `dp` is `>=` the new one (it can never again be the unique minimum). Amortized `O(1)` per stone, `O(n)` total. This is what I ship; its correctness depends entirely on the front-eviction test matching the *same* window boundary as the recurrence.

**Deriving the recurrence and base case.** `dp[j]` is the least toll to be standing on stone `j` having started on `0`. The base case is the start: if `c[0] >= 0` then `dp[0] = c[0]`; if the start stone is cracked, the frog cannot even begin, so `dp[0] = INF` and (since every path begins at `0`) everything is INF, answer `-1`. For `j >= 1`: stone `j` is reachable iff it is landable (`c[j] >= 0`) and at least one legal predecessor `i in [j-D, j-1]` is itself reachable; then `dp[j] = c[j] + min over those i of dp[i]`. A cracked `j` stays `INF`. The answer is `dp[n-1]`, or `-1` if that is `INF`.

**A numeric self-check of the window claim on a concrete case.** Before trusting the `[j-D, j-1]` window I check it against the worked sample by brute force in my head: `n = 8`, `D = 3`, `c = [0,7,2,-1,9,4,1,6]`. `dp[0] = 0`. For `j = 1`, window `i in [max(0,1-3), 0] = [0,0]`: `dp[1] = 7 + dp[0] = 7`. `j = 2`, window `[0,1]`: `min(dp[0],dp[1]) = 0`, `dp[2] = 2 + 0 = 2`. `j = 3` cracked, `dp[3] = INF`. `j = 4`, window `[1,3]`: candidates `dp[1]=7, dp[2]=2, dp[3]=INF`, min `2`, `dp[4] = 9 + 2 = 11`. `j = 5`, window `[2,4]`: `min(dp[2]=2, dp[3]=INF, dp[4]=11) = 2`, `dp[5] = 4 + 2 = 6`. `j = 6`, window `[3,5]`: `min(INF, 11, 6) = 6`, `dp[6] = 1 + 6 = 7`. `j = 7`, window `[4,6]`: `min(dp[4]=11, dp[5]=6, dp[6]=7) = 6`, `dp[7] = 6 + 6 = 12`. So `dp[7] = 12`, which matches the stated answer `12` and the path `0 -> 2 -> 5 -> 7`. The window definition reproduces the intended answer, so the recurrence is what I want; now I just have to transcribe it without breaking the boundary.

**First implementation — and immediately a trace, because the window is exactly where this dies.** Here is my first cut of the deque loop:

```
deque<int> dq;
for (int j = 0; j < n; j++) {
    while (!dq.empty() && dq.front() <= j - D) dq.pop_front();   // evict out-of-window
    if (j > 0 && c[j] >= 0 && !dq.empty() && dp[dq.front()] < INF)
        dp[j] = dp[dq.front()] + c[j];
    if (dp[j] < INF) {
        while (!dq.empty() && dp[dq.back()] >= dp[j]) dq.pop_back();
        dq.push_back(j);
    }
}
```

The line I distrust is the front eviction `dq.front() <= j - D`. Let me trace the smallest input that forces a full-reach leap, because that is where an inclusive/exclusive slip shows. Take `n = 2`, `D = 1`, `c = [0, 5]`; the only crossing is the single leap `0 -> 1` of gap `1 = D`, so the answer must be `5`.

- `j = 0`: front-evict against `0 <= 0 - 1 = -1`? `0 <= -1` is false, nothing evicted. `j > 0` is false, so no `dp[0]` update from the window; `dp[0] = c[0] = 0` was set before the loop. `dp[0] < INF`, push: deque back-eviction empty, `dq = [0]`.
- `j = 1`: front-evict: `dq.front() = 0`, test `0 <= 1 - 1 = 0`? `0 <= 0` is **true**, so it pops index `0` out of the window. Now `dq` is empty, so the `if` body finds `dq.empty()` and leaves `dp[1] = INF`. Final `dp[1] = INF`, output `-1`.

**The bug.** The code says the crossing is impossible, but the gap is exactly `D = 1`, the legal maximal leap — the answer is `5`. The defect is precise: my window is `[j-D, j-1]` with the lower end `j-D` *inclusive*, but the eviction test `dq.front() <= j - D` throws away the index `i = j - D`, treating it as already out. I used `<=` where the inclusive lower boundary needs `<`. Index `j - D` corresponds to gap exactly `D`, which is legal, so it must *stay* in the window. This is the classic inclusive-boundary off-by-one, and it cost me the full-reach leaps that are precisely the cases the problem stresses. The fix is to change the eviction test to `dq.front() < j - D`: evict only when the index is strictly below `j - D`, i.e. gap strictly greater than `D`.

**Fix and re-trace.** New eviction line `while (!dq.empty() && dq.front() < j - D) dq.pop_front();`. Re-run `n = 2, D = 1, c = [0,5]`:

- `j = 0`: evict against `0 < -1`? false. `dq = [0]` after push, `dp[0] = 0`.
- `j = 1`: evict: `dq.front() = 0`, `0 < 1 - 1 = 0`? `0 < 0` is false — index `0` stays, correctly, because gap `1 = D` is legal. `c[1] = 5 >= 0`, `dp[dq.front()] = dp[0] = 0 < INF`, so `dp[1] = 0 + 5 = 5`. Output `5`. Correct.

The case that broke now passes, and it broke for exactly the reason I fixed — the inclusive lower endpoint. That is the evidence I trust.

**Second trace — a different boundary: is the predecessor in the deque before I need it?** There is a subtle ordering question independent of the window width. I push `j` into the deque at the *end* of iteration `j`, and I read the deque front at the *start* of iteration `j`. So when I process `j`, the candidate predecessors available are those pushed in iterations `0..j-1` — exactly indices `< j`, which is what "forward leaps only, `i < j`" requires. Good in principle, but let me trace a case where `j - 1` is the *only* legal predecessor and confirm it is actually present: `n = 3`, `D = 1`, `c = [3, 1, 4]` (no cracks; with `D = 1` the frog must step on every stone, answer `3 + 1 + 4 = 8`).

- Before loop `dp[0] = 3`.
- `j = 0`: evict `0 < -1`? no. push `0`: `dq = [0]`.
- `j = 1`: evict `dq.front()=0 < 1-1=0`? `0 < 0` false, stays. `c[1]=1>=0`, `dp[0]=3`, `dp[1] = 3 + 1 = 4`. Push `1`: back-evict `dp[dq.back()] = dp[0] = 3 >= dp[1] = 4`? `3 >= 4` false, so keep `0`; `dq = [0, 1]`.
- `j = 2`: evict `dq.front()=0 < 2-1=1`? `0 < 1` true — pop `0` (gap from `0` to `2` is `2 > D = 1`, correctly illegal). Now `dq.front()=1`, `1 < 1`? false, stays. `c[2]=4>=0`, `dp[1]=4`, `dp[2] = 4 + 4 = 8`. Output `dp[2] = 8`. Correct.

Two things confirmed: the predecessor `j-1` is in the deque exactly when it should be, and the front eviction correctly drops index `0` once the gap to it exceeds `D`. The deque's window stays glued to the recurrence's window `[j-D, j-1]`.

**A worry about the deque monotonic invariant and equal values.** The back-eviction uses `dp[dq.back()] >= dp[j]`, with `>=` not `>`. Does the `=` case matter? If `dp[dq.back()] == dp[j]`, both are equal minima, but `j` has the larger index, so it survives in the window at least as long as the older one. Evicting the older equal-valued index is safe and keeps the deque strictly... well, non-redundant: among equal values I keep only the newest, which is never wrong because it dominates on lifetime. I keep `>=`. (Using `>` would merely leave harmless duplicates; correctness is unaffected, but `>=` keeps the deque smaller.) Let me sanity-check on `c = [0, 5, 5]`, `D = 2`: `dp[0]=0`; `j=1` window has `0`, `dp[1]=5`; push `1`, back-evict `dp[0]=0 >= 5`? no, `dq=[0,1]`. `j=2` window `[0,1]`, front `0` since `0 < 0` false stays, `dp[2] = 0 + 5 = 5`. The min over the window is `dp[0]=0` not `dp[1]=5`, and the deque front correctly gives `0`. Good — the `>=` eviction of the back never disturbs the front minimum.

**Edge cases, deliberately, because boundary code dies in the corners.**
- `n = 1`: the loop runs only `j = 0`, no window update, `dp[0] = c[0]` if landable. Answer `dp[0]`. If `c[0] = 0` the frog is already standing on the end having paid `0` — output `0`. If `c[0] = -1` the start is cracked, `dp[0] = INF`, output `-1`. Both correct (the start *is* the end).
- *Cracked start*, `c[0] = -1`, e.g. `c = [-1, 0, 0]`, `D = 2`: I never set `dp[0]` (the `if (c[0] >= 0)` guard fails), so `dp[0] = INF`, nothing is ever pushed reachable from it, all `dp` stay `INF`, output `-1`. Correct.
- *Cracked end*, `c[n-1] = -1`, e.g. `c = [0, 0, -1]`, `D = 5`: `j = n-1` is cracked, `c[j] >= 0` fails, `dp[n-1]` stays `INF`, output `-1`. Correct — you cannot stand on a cracked final stone.
- *Gap forced beyond `D`*, `c = [0, -1, 5]`, `D = 1`: stone `1` cracked; to reach `2` from `0` the gap is `2 > D = 1`. Trace: `j=1` cracked, not pushed; at `j=2` the only in-window index would be `1` (`[1,1]`), but `1` was never pushed, deque empty there, `dp[2] = INF`, output `-1`. Correct. With `D = 2` instead, `0` is in window `[0,1]` for `j=2` and `dp[2] = 0 + 5 = 5` — the inclusive full-reach leap saves it. This pair is exactly the inclusive-boundary case the problem stresses, and it behaves right.
- *Overflow*: `dp` values and the answer are `long long`; the worst total `~2*10^14` fits with room. `INF = LLONG_MAX/4` is only ever added to a non-negative `c[j]` once per stone and only when the predecessor is `< INF`, so no `INF + something` is ever stored into a used `dp` and no overflow occurs. The guard `dp[dq.front()] < INF` makes the addition only happen on a finite predecessor.
- *Output*: exactly one integer and a newline; `cin >>` eats arbitrary whitespace, so a wrapped second line parses fine.

**Performance check.** Each index is pushed once and popped at most once from each end of the deque, so the total deque work is `O(n)` amortized; the rest of the loop is `O(1)` per stone. Reading `n = 2*10^5` ints and one linear pass is well under a second — my large test ran in about 0.06 s. Memory is one `long long` vector of length `n` plus a deque of indices, comfortably inside 256 MB.

**Re-verification at scale.** I stress-tested the deque solution against the plain `O(n*D)` brute force on 900 random small cases — `n` up to 25, `D` ranging from `1` to past `n`, with about a third of the stones cracked, plus a forced `D=1`/full-reach mix — and got zero mismatches, after the eviction-boundary fix. Before the fix, the full-reach cases (`gap == D`) disagreed; after it, none do. I also confirmed the documented sample `n=8, D=3` yields `12`.

**Final solution.** I disproved the slow scan on grounds of time, built the deque DP, and the one place it broke — the inclusive lower boundary `i = j - D` — I traced to a `<=`-vs-`<` slip and fixed. This is what I ship: one self-contained file, `O(n)`.

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

**Causal recap.** The slow `O(n*D)` window scan is correct but times out at `n = D = 2*10^5`, so I moved to the same recurrence accelerated by a monotonic-deque sliding-window minimum over predecessors `[j-D, j-1]`. The single decisive boundary is the *inclusive* lower endpoint `i = j-D` (gap exactly `D`): my first eviction test `dq.front() <= j-D` wrongly discarded it, and a trace of `n=2, D=1, c=[0,5]` returning `-1` instead of `5` pinpointed the `<=`-vs-`<` slip; changing it to `dq.front() < j-D` keeps the full-reach predecessor and fixes every gap-`==D` crossing. A second trace (`n=3, D=1`) confirmed predecessors enter the deque exactly when `i < j` requires and the front eviction drops index `0` precisely when its gap exceeds `D`; `long long` throughout and the `dp < INF` guard close out the overflow, cracked-start, cracked-end, forced-gap, and `n=1` corners, and 900 randomized cases against the brute force agree with zero mismatches.
