**Problem.** Given `n` weighted intervals, where interval `i` is the half-open segment `[s_i, e_i)`
(touching endpoints do not overlap) with positive weight `w_i`, choose a pairwise non-overlapping
subset maximizing the total weight; the empty set is allowed, so the answer is at least `0`. Read
`n` and the `n` triples `s e w` from stdin, print the maximum total weight.

**Why the obvious greedy is wrong.** The famous earliest-finish-time greedy — sort by end, sweep,
take any interval that does not overlap the last one taken — is *optimal for the unweighted* problem
(most intervals possible), but it optimizes the **count**, not the **weight**. On
`{[0,1) w=1, [0,100) w=1000}` it takes the early-finishing `[0,1)` for `1` and is then blocked from
`[0,100)`, missing the optimum `1000`. On `{[0,2) w=1, [1,3) w=1, [2,4) w=1, [0,4) w=5}` it collects
two unit intervals (`A`,`C`) for `2` instead of the single heavy `[0,4)` worth `5`. Earliest-finish
greedy is discarded.

**Key idea — sort-by-end DP with binary-search predecessor.** Sort the intervals by end coordinate
so `e_0 <= e_1 <= ... <= e_{n-1}`. Process them in that order, maintaining

- `best[k]` = max total weight using only the first `k` intervals (`best[0] = 0`).

For interval `i` (0-indexed in sorted order):

- *Skip `i`:* value `best[i]`.
- *Take `i`:* value `w_i + best[p(i)]`, where `p(i)` is the number of earlier intervals that finish
  at or before `s_i`, i.e. `e_j <= s_i` (the `<=` is the half-open touching rule). Because the
  intervals are sorted by end, those compatible predecessors are exactly the prefix `[0, p(i))`, so
  appending `i` to *any* solution counted by `best[p(i)]` stays non-overlapping — that is the
  correctness guarantee greedy lacked.

So `best[i+1] = max(best[i], w_i + best[p(i)])`, and the answer is `best[n]`. Since the ends are
sorted, `p(i)` is found by binary search (the count of ends `<= s_i`), giving `O(n log n)` overall.

**Two pitfalls to get right.**
1. *Predecessor boundary.* Compute `p(i)` with a single half-open `[lo, hi)` binary search that
   returns the **count** of ends `<= s_i`; that count *is* the index into `best[]` (no `+1`/`-1`
   fixup). Mixing a "last index" reading with a "count" reading silently drops one predecessor's
   contribution on non-trivial prefixes. (A trace of `[0,2)w5 / [1,4)w1 / [2,6)w5` expecting `10`,
   plus oracle differential testing, exposes this.)
2. *Overflow.* Coordinates reach `2*10^9` (already past 32-bit `int`) and the running total reaches
   `n * w = 2*10^5 * 10^9 = 2*10^14`; store `s, e, w` and the DP table as `long long`. An `int` is a
   silent wrong-answer on the large tests.

**Edge cases (handled by the recurrence + the `<=` test):** `n = 0` -> `0`; a single interval ->
its own weight; intervals touching at a point -> both selectable; mutually overlapping duplicates ->
only one taken; nested intervals -> the better of the outer interval versus the disjoint inner chain.

**Complexity.** `O(n log n)` time (one sort plus one binary search per interval), `O(n)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0

    struct Job { long long s, e, w; };
    vector<Job> job(n);
    for (auto &j : job) cin >> j.s >> j.e >> j.w;

    // Sort by finishing time (end coordinate), ascending.
    sort(job.begin(), job.end(),
         [](const Job &x, const Job &y) { return x.e < y.e; });

    // ends[i] = finishing time of the i-th job in sorted order.
    vector<long long> ends(n);
    for (int i = 0; i < n; i++) ends[i] = job[i].e;

    // best[i] = max total weight achievable using only jobs[0..i].
    // Intervals are half-open [s, e): job j is compatible with job i (j before i)
    // iff ends[j] <= starts[i]. p(i) = largest index j < i with ends[j] <= job[i].s.
    vector<long long> best(n + 1, 0); // best[0] = 0 (no jobs)
    for (int i = 0; i < n; i++) {
        // Skip job i: best[i] (using jobs[0..i-1]).
        long long skip = best[i];
        // Take job i: its weight plus best over jobs ending at or before job[i].s.
        // Find p = number of jobs (in sorted prefix [0..i-1]) whose end <= job[i].s.
        int lo = 0, hi = i; // search in ends[0..i-1]
        long long key = job[i].s;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (ends[mid] <= key) lo = mid + 1;
            else hi = mid;
        }
        // lo = count of indices in [0..i-1] with ends[] <= key, i.e. p(i)+1 in 1-based best[].
        long long take = job[i].w + best[lo];
        best[i + 1] = max(skip, take);
    }

    cout << best[n] << "\n";
    return 0;
}
```
