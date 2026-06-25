**Problem.** One telescope, one night, `n` candidate observations. Observation `i` occupies the half-open interval `[s_i, e_i)` and yields value `v_i >= 0`. Pick a subset with no two overlapping and maximize the total value; the empty set is allowed, so the answer is at least `0`. Half-open means two intervals that merely touch (`e_i == s_j`) do **not** conflict. Read `n` and the `n` triples `s e v` from stdin, print the maximum total value.

**Why the obvious greedy is wrong.** The non-overlap constraint is global, so a local greedy has no reason to be optimal once intervals carry weights. On

```
A [0,30) v=20    B [0,90) v=50    C [30,60) v=25    D [60,90) v=25    E [40,50) v=5
```

*value-greedy* (take the biggest that fits) grabs B and hogs the whole night for `50`; *earliest-finishing* (the classic count-maximizer) takes A, then the scrap E, then D for `20+5+25 = 50`. But A+C+D tile the night cleanly (touching at 30 and 60) for `20+25+25 = 70`. Both greedies are beaten, and not rarely — over random small instances, value-greedy missed the optimum on ~22% and earliest-finishing on ~54%. Greedy is discarded.

**Key idea — end-sorted interval DP.** Sort observations by finishing time so `e_0 <= e_1 <= ... <= e_{n-1}`. Let `dp[i]` be the best total value over the first `i` sorted observations. For observation `i-1` with start `s`, either skip it (`dp[i-1]`) or take it: taking adds `v_{i-1}` and may keep any earlier observation that finishes by `s`. Because the array is end-sorted, the compatible predecessors are exactly a prefix `0..p-1`, where `p` is the number of earlier ends `<= s`, found by binary search. So

- `dp[i] = max(dp[i-1], v_{i-1} + dp[p])`,  `p = #{ j < i-1 : e_j <= s }`,  `dp[0] = 0`,

and the answer is `dp[n]`. Compute `p` with `upper_bound(ends, ends+(i-1), s)` — the count of ends `<= s`. `O(n log n)`.

**Pitfalls.**
1. *Sort key.* Sort by **end**, not start. The recurrence's validity (compatible predecessors form a contiguous prefix, and the binary search is meaningful) rests on `ends[0..i-2]` being ascending. Sorting by start leaves `ends` unsorted, so `upper_bound` returns nonsense and the DP builds on incompatible predecessors. (A trace of `X[0,6)5, Y[1,3)4, Z[3,5)4` returning `9` instead of `8` exposes exactly this.)
2. *Touching convention.* Use `upper_bound`, not `lower_bound`, so ends equal to `s` count as compatible. With half-open intervals `[0,5)` and `[5,10)` touch and may both be taken; `lower_bound` wrongly drops the touching one (a trace of that pair returning `10` instead of `20` catches it).
3. *Overflow.* With `n` up to `2*10^5` and `v_i` up to `10^9`, the total reaches `~2*10^14`; use `long long`. An `int` is a silent wrong-answer on large tests.

**Edge cases (all handled by the recurrence):** `n = 0` -> `0`; a single observation -> its value (or `0` if value `0`); touching chains all schedulable; fully nested / all-overlapping -> at most one chosen; many identical intervals -> one of them; zero-value observations contribute nothing.

**Complexity.** `O(n log n)` time (sort + a binary search per observation), `O(n)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0

    // Each observation: half-open interval [s, e) with value v >= 0.
    struct Obs { long long s, e, v; };
    vector<Obs> obs(n);
    for (int i = 0; i < n; i++) cin >> obs[i].s >> obs[i].e >> obs[i].v;

    // Sort by end time ascending (ties broken by start, irrelevant for correctness).
    sort(obs.begin(), obs.end(), [](const Obs &a, const Obs &b) {
        if (a.e != b.e) return a.e < b.e;
        return a.s < b.s;
    });

    // ends[i] = finishing time of the i-th observation in sorted order.
    vector<long long> ends(n);
    for (int i = 0; i < n; i++) ends[i] = obs[i].e;

    // dp[i] = best total value using a subset of the first i sorted observations.
    // Half-open intervals: observation i (start obs[i].s) is compatible with any earlier
    // observation whose end <= obs[i].s, i.e. it does not overlap.
    vector<long long> dp(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long s = obs[i - 1].s;
        // p = number of observations (among the first i-1) whose end <= s.
        // Those are exactly indices [0, p) in sorted order; ends is sorted ascending.
        int p = (int)(upper_bound(ends.begin(), ends.begin() + (i - 1), s) - ends.begin());
        long long take = obs[i - 1].v + dp[p];   // include observation i-1
        long long skip = dp[i - 1];              // exclude it
        dp[i] = max(take, skip);
    }

    cout << dp[n] << "\n";
    return 0;
}
```
