# Weighted interval scheduling: maximum-weight non-overlapping selection

## Research question

You are given `n` intervals. Interval `i` is the half-open segment `[s_i, e_i)` (it occupies
every point `x` with `s_i <= x < e_i`) and carries a positive weight `w_i`. Two intervals
**overlap** if they share an interior point; because the segments are half-open, intervals that
merely touch at an endpoint (one ends exactly where another begins, `e_i = s_j`) do **not**
overlap and may both be selected.

Choose a subset of the intervals that are pairwise non-overlapping (you may choose none) so that
the **sum of the chosen weights is maximized**. Output that maximum total weight. Because the empty
subset is allowed, the answer is always at least `0`.

This is *weighted* interval scheduling. Its unweighted cousin — "select as many pairwise
non-overlapping intervals as possible" — has a famous one-line greedy answer (repeatedly take the
interval that finishes earliest). The weighted version asks for the most *valuable* compatible set,
not the most *numerous* one.

## Input / output contract

- Input (stdin):
  - The first token is `n` (`0 <= n <= 2*10^5`).
  - Then `n` triples follow, each `s_i e_i w_i`, whitespace-separated:
    - `0 <= s_i < e_i <= 2*10^9` (so each interval is non-empty),
    - `1 <= w_i <= 10^9`.
- Output (stdout): a single line with the maximum achievable total weight.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for the four intervals

```
[0, 5) w=5
[1, 6) w=3
[5, 9) w=6
[6, 10) w=4
```

the answer is `11`: take `[0, 5)` and `[5, 9)` (they touch at `5`, so they are compatible) for
`5 + 6 = 11`. No compatible pair beats this.

## Evaluation settings

Judged on hidden tests covering: the empty instance (`n = 0`); a single interval; intervals that
touch at endpoints (must be allowed together); heavily overlapping clusters; nested intervals;
duplicate intervals; and large `n = 2*10^5` with coordinates and weights near their maxima.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    struct Job { long long s, e, w; };
    vector<Job> job(n);
    for (auto &j : job) cin >> j.s >> j.e >> j.w;

    // TODO: choose a pairwise non-overlapping subset of the intervals [s, e)
    // (touching endpoints allowed) maximizing the total weight; empty subset -> 0.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
