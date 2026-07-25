# Lattice points covered by a union of axis-aligned rectangles

## Research question

You are given `n` axis-aligned rectangles on the integer plane. Each rectangle is described by two
opposite corners with integer coordinates; the rectangle is **closed**, meaning it covers every
integer lattice point that lies inside it **or on its boundary** (the four edges and the four
corners are included). Rectangles may overlap, touch, or be degenerate (a single point or a thin
line segment). Count the number of **distinct** integer lattice points that are covered by **at
least one** rectangle, and output that count.

This is the lattice-point (point-counting) cousin of the classic "area of a union of rectangles"
problem. The twist is that the answer is a *count of grid points*, not an area, so every dimension
carries an inclusive `+1`: a rectangle spanning columns `x1..x2` covers `x2 - x1 + 1` columns, not
`x2 - x1`. Getting that inclusive/exclusive boundary exactly right — at the level of a single
rectangle, at shared edges between rectangles, and at degenerate rectangles — is the entire
difficulty.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`). Then `n` lines (or just `4*n`
  whitespace-separated tokens) follow, each giving four integers `a b c d` with
  `-10^9 <= a, b, c, d <= 10^9`. The corners are `(a, b)` and `(c, d)`; they are *opposite* corners
  but given in **arbitrary** order, so it is **not** guaranteed that `a <= c` or `b <= d`. The
  rectangle has edges parallel to the axes through these two corners.
- Output (stdout): a single line with the number of distinct integer lattice points covered by the
  union of the `n` closed rectangles.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for the two rectangles `(0,0)-(2,2)` and `(1,1)-(4,3)` the answer is `17`. The first
covers the 9 points of the 3x3 block `x in {0,1,2}, y in {0,1,2}`; the second covers the 12 points
of the 4x3 block `x in {1,2,3,4}, y in {1,2,3}`; they share the 4 points `x in {1,2}, y in {1,2}`,
so the union has `9 + 12 - 4 = 17` distinct lattice points.

## Evaluation settings

Judged on hidden tests covering: a single rectangle (including a single point `a=c, b=d` and a thin
horizontal/vertical line); the empty input `n = 0` (answer `0`); rectangles given with corners in
reversed order; rectangles that overlap heavily, that touch along a shared edge, and that touch only
at a single corner (the shared edge/corner points must be counted exactly once); and large cases
with many rectangles spanning the full coordinate range.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    for (int i = 0; i < n; i++) {
        long long a, b, c, d;
        cin >> a >> b >> c >> d; // opposite corners, arbitrary order
        // normalize to x1<=x2, y1<=y2 ...
    }

    // TODO: count distinct integer lattice points covered by the union of the closed rectangles.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
