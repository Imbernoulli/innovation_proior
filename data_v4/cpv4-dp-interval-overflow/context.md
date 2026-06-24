# Cheapest triangulation of a labelled convex polygon

## Research question

You are given a convex polygon with `n` vertices, listed in boundary order, where vertex `i` carries a
positive integer label `v[i]`. A **triangulation** cuts the polygon into `n - 2` triangles using
`n - 3` non-crossing diagonals between vertices. Each resulting triangle, with corner vertices `a`,
`b`, `c`, costs `v[a] * v[b] * v[c]` (the product of its three labels). Choose a triangulation that
**minimizes the total cost** (the sum of the per-triangle products) and output that minimum.

If `n < 3` there are no triangles, so the cost is `0`.

This is the classic interval-DP shape (matrix-chain / optimal-triangulation family): the optimum over a
boundary arc decomposes into the optimum over two shorter arcs joined by a single triangle. Getting it
exactly right means nailing the arc recurrence, the base case for an edge, and — because the costs are
*products* — the arithmetic width of every value that gets summed.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 500`); then `n` integers `v[i]`
  (`1 <= v[i] <= 2000`), whitespace-separated, in convex-boundary order.
- Output (stdout): a single line with the minimum total triangulation cost.
- Time limit: 1 second. Memory: 256 MB.

Example: for the square `v = [10, 20, 30, 40]` the answer is `18000`. The diagonal `0–2` gives triangles
`(0,1,2) = 6000` and `(0,2,3) = 12000`, totalling `18000`; the other diagonal `1–3` totals `32000`, so
`18000` is optimal.

## Background

The constraint is that diagonals may not cross, so a triangulation is *nested*, which is exactly what
makes an interval DP natural. Two routes are worth weighing before committing:

- **Greedy / fan from a vertex.** Triangulate by drawing every diagonal from one fixed vertex (a "fan"),
  perhaps the smallest-labelled one to keep its factor in every triangle small. It is `O(n)` and trivial.
  The open question is whether any single fixed fan can be optimal, given that the best triangulation may
  need diagonals that share no common vertex.
- **Interval dynamic programming.** For every contiguous boundary arc `i..j`, compute the cheapest way to
  triangulate the sub-polygon bounded by that arc and the chord `i–j`, by choosing the apex `k` of the
  triangle that uses the chord `i–j`. This is `O(n^3)` over `O(n^2)` states. The open questions are the
  exact recurrence, the edge base case, and the integer width of the accumulated products.

## Evaluation settings

Judged on hidden tests covering: tiny polygons (`n = 0, 1, 2, 3`), small polygons checked against a full
triangulation enumeration, polygons whose optimum needs crossing-free diagonals that no single fan can
reproduce, and large polygons (`n = 500`) with labels near `2000`. With labels up to `2000`, a single
triangle already costs up to `2000^3 = 8 * 10^9`, which overflows a 32-bit integer, and the summed answer
can reach roughly `4 * 10^12` — so every product and every accumulator must be 64-bit.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> v(n);
    for (auto &x : v) cin >> x;

    if (n < 3) { cout << 0 << "\n"; return 0; }

    // TODO: interval DP over boundary arcs i..j; for arc (i, j) pick the apex k
    // of the triangle on chord (i, j) and add its product cost v[i]*v[k]*v[j].
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
