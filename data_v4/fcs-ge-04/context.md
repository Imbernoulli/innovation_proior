# Interior lattice points of a simple polygon

## Research question

You are given a **simple polygon** (a closed, non-self-intersecting polygon) with `n` vertices, each
having **integer** coordinates, listed in order around the boundary (either clockwise or
counter-clockwise). Count the number of lattice points (points with integer coordinates) that lie
**strictly inside** the polygon — points on the boundary do **not** count.

Concretely: how many integer points `(x, y)` are enclosed by the polygon but not on any of its edges
or vertices?

This is the integer-geometry core that turns up inside area-counting, triangulation, and
computational-geometry pipelines, so getting it exact — with no floating point and no overflow — is
what matters.

## Input / output contract

- Input (stdin):
  - The first token is `n` (`3 <= n <= 10^5`), the number of polygon vertices.
  - Then `n` lines (or whitespace-separated pairs) each give two integers `x_i y_i`
    (`-10^9 <= x_i, y_i <= 10^9`), the vertices in boundary order.
  - The polygon is guaranteed **simple** (edges do not cross or touch except at shared endpoints) and
    **non-degenerate** (its area is positive; the vertices are not all collinear).
- Output (stdout): a single line with the number of lattice points strictly interior to the polygon.
- Time limit: 1 second. Memory: 256 MB.

Example: for the L-shaped hexagon `(0,0), (4,0), (4,2), (2,2), (2,4), (0,4)` the answer is `5`.

## Evaluation settings

Judged on hidden tests covering: convex and concave (non-convex) polygons; polygons given clockwise and
counter-clockwise; triangles and many-vertex polygons; polygons with collinear vertices along an edge;
thin slivers with zero interior points; polygons placed at large negative and positive coordinates near
`10^9`; and `n` up to `10^5`.

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
    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];

    // TODO: count lattice points strictly inside the simple polygon, in O(n),
    //       using exact integer arithmetic (no floating point, no overflow).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
