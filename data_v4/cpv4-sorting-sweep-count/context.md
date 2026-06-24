# Counting close pairs on a circular track

## Research question

There are `n` runners standing on a circular track of integer circumference `L`. Runner `i` stands at
position `p[i]` measured clockwise from a fixed start line, with `0 <= p[i] < L`. Several runners may
share a position. The **circular distance** between two runners is the shorter of the two arcs joining
them: if `d = |p[i] - p[j]|`, the circular distance is `min(d, L - d)`.

Count the number of **unordered pairs** of runners whose circular distance is at most `D`. Output that
count.

This is a sorting-and-sweep counting problem. The twist is the wrap-around: two runners are "close"
either by the short way *or* by going around the back of the track, and the back-of-the-track case is
exactly where a counting sweep tends to double-count or miss the regime where *every* pair qualifies.

## Input / output contract

- Input (stdin): the first line has three integers `n`, `L`, `D`
  (`0 <= n <= 2*10^5`, `1 <= L <= 10^9`, `0 <= D <= L`).
  The second line has `n` integers `p[i]` (`0 <= p[i] < L`), whitespace-separated. When `n = 0` the
  second line is empty or absent.
- Output (stdout): a single line with the number of unordered pairs `{i, j}` (`i != j`) whose circular
  distance `min(|p[i]-p[j]|, L-|p[i]-p[j]|)` is `<= D`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 4`, `L = 10`, `D = 2`, positions `p = [0, 1, 5, 9]`, the answer is `3`: the pairs
`{0,1}` (distance 1), `{0,9}` (distance `min(9,1)=1`, around the back), and `{1,9}` (distance
`min(8,2)=2`, around the back). The other three pairs sit at distance 4 or 5.

## Background

After sorting the positions, "close pairs" are usually counted with a two-pointer sweep: for each right
endpoint, advance a left pointer to the first position still within the window, and add the window
width. The circular metric forces two refinements before that template is correct:

- **The metric splits into two linear conditions.** With both positions in `[0, L)`, the raw gap is
  `d = |p[i]-p[j]| in [0, L-1]`. Circular distance `<= D` means `d <= D` (close the short way) **or**
  `L - d <= D`, i.e. `d >= L - D` (close the long way). So a pair qualifies iff `d <= D` or
  `d >= L - D`.
- **The two conditions can overlap.** When `2*D >= L` the threshold `L - D` drops at or below `D`, so
  the two intervals of qualifying gaps meet or overlap and *every* pair qualifies. Counting "short" and
  "long" pairs separately and adding them would then double-count.

Two families of approach are on the table:

- **Direct min-distance sweep.** Compute `min(d, L-d)` per pair and count — but a single window over the
  sorted array only captures the short side; the wrap-around pairs need separate handling, and it is
  easy to conflate the two.
- **Split-by-regime sweep.** Handle `2*D >= L` as "all pairs" up front; otherwise the short and long
  conditions are disjoint, so run two independent two-pointer sweeps and add. The open question is the
  exact pointer comparisons (`<=` vs `<`, `>=` vs `>`) and the per-endpoint increment.

## Evaluation settings

Judged on hidden tests covering: `D = 0` (only coincident pairs count), `2*D >= L` (every pair counts,
including `D = L`), heavy duplicate positions, `n = 0` and `n = 1` (answer `0`), all runners at the same
spot, and large `n = 2*10^5` with `L` near `10^9` so the pair count exceeds a 32-bit integer.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L, D;
    if (!(cin >> n >> L >> D)) return 0;
    vector<long long> p(n);
    for (auto &x : p) cin >> x;
    sort(p.begin(), p.end());

    // TODO: count unordered pairs whose circular distance min(|pi-pj|, L-|pi-pj|) is <= D.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
