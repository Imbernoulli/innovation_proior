**Problem.** A row of `m` vents (`0..m-1`, `m <= 20`) must all be sealed. There are `n` strips; strip
`i` seals the **half-open** span `[a_i, b_i)` — vents `a_i .. b_i-1` — at cost `c_i`. Strips may
overlap and any subset may be applied. Read `m`, `n`, and the `n` triples from stdin; print the
minimum total cost to seal every vent, or `-1` if impossible.

**Why the obvious greedy is wrong.** "Sweep left to right, take the strip reaching furthest right"
optimizes coverage *length*, but cost is per strip and overlaps are free. On `m = 4` with a wide strip
`{0,1,2,3}` at cost `12` versus two narrow strips `{0,1}@5` and `{2,3}@6`, greedy grabs the wide one
for `12` while `5 + 6 = 11` is cheaper. Greedy is discarded.

**Key idea — subset (bitmask) DP.** The union you must build is an arbitrary subset of the `m` vents,
so the state is *which vents are sealed*, a bitmask. Let `dp[S]` = minimum cost to have sealed exactly
the set `S`. Initialize `dp[0] = 0`, everything else `+infinity`. Relax by adding one strip at a time:

- `dp[S | mask_i] = min(dp[S | mask_i], dp[S] + c_i)` for every strip `i`, scanning `S = 0 .. 2^m-1`.

The answer is `dp[(1<<m)-1]`, or `-1` if it stays infinite. A single forward pass over `S` suffices
because every transition moves to a strict superset `nS = S | mask_i > S`, so `dp[nS]` is only relaxed
from already-finalized lower indices.

**Pitfalls to get right.**
1. *The half-open boundary (the decisive off-by-one).* `[a, b)` seals vents `a..b-1`. Build the mask
   with `for (h = a; h < b; h++) bits |= 1<<h` — inclusive `a`, **exclusive** `b`. Writing `h <= b`
   seals one phantom vent on the right: it sets a bit for a vent the strip does not cover (and even a
   nonexistent bit `>= m`), and it makes strips that should tile disjointly (`[0,2)` and `[2,4)`)
   overlap. The symptom is brutal: a perfectly coverable instance prints `-1`, because no strip
   combination ever produces the exact full mask. A trace of `0 2` + `2 4` at `m=4` exposes it
   immediately — the fix is the single character `<` instead of `<=`.
2. *Cost type / overflow.* With `n` up to `2*10^5` and `c_i` up to `10^9`, costs exceed 32 bits; use
   `long long`. Keep `INF = 4e18` (below `LLONG_MAX`) and guard `if (dp[S] == INF) continue;` so an
   `INF` cell is never used as a relaxation base — no `INF + cost` wrap. Test `dp[FULL] >= INF` for
   the impossible case.

**Edge cases.** `m = 1` with one covering strip -> its cost; a vent in no strip's span -> `-1`;
zero-cost strips -> handled (`dp[S] + 0` is a valid relax, free covers allowed); touching spans
`[0,2)` and `[2,4)` -> disjoint after the `< b` fix.

**Complexity.** `O(2^m * n)` time, `O(2^m)` memory. At `m = 20` the `2^m` factor dominates; distinct
contiguous spans number at most `m(m+1)/2 <= 210`, so collapsing duplicate masks to their cheapest
representative bounds the inner loop and keeps large `n` within the 2 s limit.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m, n;
    if (!(cin >> m >> n)) return 0;            // m vents (0..m-1), n sealant strips

    const long long INF = (long long)4e18;
    // Each strip seals a contiguous span of vents given as [a, b) (half-open: a..b-1) at cost c.
    // Build the bitmask of vents a strip covers, then run a set-cover DP over 2^m subsets.
    vector<int> mask(n);
    vector<long long> cost(n);
    for (int i = 0; i < n; i++) {
        int a, b; long long c;
        cin >> a >> b >> c;                    // strip seals vents a..b-1, cost c
        int bits = 0;
        for (int h = a; h < b; h++) bits |= (1 << h);   // inclusive a, exclusive b
        mask[i] = bits;
        cost[i] = c;
    }

    int FULL = (1 << m) - 1;
    // dp[S] = minimum total cost of a multiset of strips whose union of sealed vents is exactly S
    //         reachable by adding strips one at a time. We only need to reach the FULL set.
    vector<long long> dp(1 << m, INF);
    dp[0] = 0;
    for (int S = 0; S < (1 << m); S++) {
        if (dp[S] == INF) continue;
        for (int i = 0; i < n; i++) {
            int nS = S | mask[i];
            if (dp[S] + cost[i] < dp[nS]) dp[nS] = dp[S] + cost[i];
        }
    }

    if (dp[FULL] >= INF) cout << -1 << "\n";
    else cout << dp[FULL] << "\n";
    return 0;
}
```
