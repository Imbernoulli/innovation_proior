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
left-to-right scheduling where consecutive picks must differ, so getting the
one-dimensional version exactly right — including the single-color and
empty-row corners and the 64-bit cost accumulation — matters.

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
cannot be chosen in isolation. Two families of approach are on the table before
committing to one:

- **Greedy.** Walk the row left to right (or pick the globally cheapest cells
  first) and take the cheapest color that does not clash with an already-fixed
  neighbour. This is near-trivial to write and `O(nk)` or `O(nk log(nk))`; the
  open question is whether locally cheapest choices can be forced into a globally
  expensive coloring.
- **Layered dynamic programming.** For each house keep, per color, the best total
  cost of a valid coloring of the prefix that ends with the house painted that
  color. The transition for house `i`, color `c` adds `cost[i][c]` to the
  cheapest entry of the previous house among colors `!= c`. The open questions are
  the exact recurrence, how to compute "cheapest previous entry of a different
  color" without an `O(k)` inner scan per color (which would be `O(nk^2)`), and
  the first-house / single-color corners.

## Evaluation settings

Judged on hidden tests covering: tiny rows (`n <= 8`, small `k`) checked against
exhaustive enumeration; adversarial "one cheap color everywhere" rows that defeat
greedy; `k = 1` with `n >= 2` (impossible, `-1`) and `k = 1` with `n <= 1`
(trivial); the empty row (`n = 0`, answer `0`); `k = 2`; uniform-cost rows; and
large rows `n = 10^5`, `k = 100` with costs near `10^9` (so the total can reach
about `10^{14}`, exceeding 32-bit range, and an `O(nk^2)` method would be too
slow).

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
