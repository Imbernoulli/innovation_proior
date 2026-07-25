# Minimum cost to paint a row of houses, no two adjacent the same color

## Research question

There is a row of `n` houses. Each house must be painted with exactly one of `k`
colors. Painting house `i` with color `c` costs `cost[i][c]`. The only constraint
is that **no two adjacent houses may share a color**. Choose a color for every
house so that the **total painting cost is minimized**, and output that minimum.

If it is impossible to color the row under the adjacency rule (this happens only
when `k = 1` and `n >= 2`, where the single color would force two neighbours to
match), output `-1`.

This is the linear "list coloring with per-cell weights" problem. The same shape
shows up inside sequence labeling, segment/state-assignment DPs, and any
left-to-right scheduling where consecutive picks must differ.

## Input / output contract

- Input (stdin):
  - The first line contains two integers `n` and `k`
    (`0 <= n <= 10^5`, `1 <= k <= 100`).
  - Then `n` lines follow; line `i` (0-indexed) contains `k` integers
    `cost[i][0..k-1]` (`0 <= cost[i][c] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the minimum total cost, or `-1` if no
  valid coloring exists.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `n = 3`, `k = 3` and costs

```
17 2 17
16 16 5
14 3 19
```

the answer is `10` (paint house 0 green = 2, house 1 blue = 5, house 2 green = 3;
adjacent houses differ, and `2 + 5 + 3 = 10` is minimal).

## Background

The adjacency rule couples consecutive houses, so each house's cheapest color
cannot be chosen in isolation.

## Evaluation settings

Judged on hidden tests covering: tiny rows (`n <= 8`, small `k`) checked against
exhaustive enumeration; adversarial rows constructed to defeat naive greedy
heuristics; `k = 1` with `n >= 2` and `k = 1` with `n <= 1`; the empty row
(`n = 0`); `k = 2`; uniform-cost rows; and large rows `n = 10^5`, `k = 100` with
costs near `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, k;
    if (!(cin >> n >> k)) return 0;

    // Read cost[i][c] for i in [0,n), c in [0,k) and compute the minimum total
    // cost of painting all houses so that no two adjacent houses share a color.
    // Output -1 if no valid coloring exists (only when k == 1 and n >= 2).

    // TODO: compute the minimum total painting cost (or -1 if impossible).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
