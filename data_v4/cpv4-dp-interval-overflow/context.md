# Cheapest triangulation of a labelled convex polygon

## Research question

You are given a convex polygon with `n` vertices, listed in boundary order, where vertex `i` carries a
positive integer label `v[i]`. A **triangulation** cuts the polygon into `n - 2` triangles using
`n - 3` non-crossing diagonals between vertices. Each resulting triangle, with corner vertices `a`,
`b`, `c`, costs `v[a] * v[b] * v[c]` (the product of its three labels). Choose a triangulation that
**minimizes the total cost** (the sum of the per-triangle products) and output that minimum.

If `n < 3` there are no triangles, so the cost is `0`.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 500`); then `n` integers `v[i]`
  (`1 <= v[i] <= 2000`), whitespace-separated, in convex-boundary order.
- Output (stdout): a single line with the minimum total triangulation cost.
- Time limit: 1 second. Memory: 256 MB.

Example: for the square `v = [10, 20, 30, 40]` the answer is `18000`. The diagonal `0–2` gives triangles
`(0,1,2) = 6000` and `(0,2,3) = 12000`, totalling `18000`; the other diagonal `1–3` totals `32000`, so
`18000` is optimal.

## Evaluation settings

Judged on hidden tests covering: tiny polygons (`n = 0, 1, 2, 3`), small polygons checked against a full
triangulation enumeration, polygons whose optimum needs crossing-free diagonals that no single fan can
reproduce, and large polygons (`n = 500`) with labels near `2000`.

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

    // TODO: compute the minimum total triangulation cost, where a triangle with
    // corners a, b, c costs v[a]*v[b]*v[c].
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
