# Maximum-area triangle over a set of planar points

## Research question

You are given `n` points with integer coordinates in the plane. Choose three of them to form a
triangle of **maximum area**. Output **twice** that maximum area (i.e. the maximum, over all triples,
of the absolute value of the cross product of two triangle edges). Twice the area is always a
non-negative integer for integer coordinates, so the answer is exact — no floating point is involved.

If no triangle has positive area (fewer than three points, or all points collinear), the answer is `0`.

This is the planar *maximum-area inscribed triangle* problem. It looks like a textbook "convex hull +
rotating calipers" exercise, but it carries a famous trap: the long-believed `O(n)` rotating-calipers
algorithm for it (Dobkin–Snyder, 1979) is **wrong**, so the contract above must be met by an algorithm
that is actually correct at this scale.

## Input / output contract

- Input (stdin):
  - The first token is `n` (`0 <= n <= 5000`).
  - Then `n` pairs of integers `x[i] y[i]` (`-10^9 <= x[i], y[i] <= 10^9`), whitespace-separated.
  - Points may repeat and may be collinear.
- Output (stdout): a single line with one integer — twice the maximum triangle area (`0` if no
  positive-area triangle exists).
- Time limit: 1 second. Memory: 256 MB.

Example: for the 6 points `(0,0) (4,0) (4,3) (2,5) (0,4) (2,2)` the answer is `20` (the triangle
`(0,0),(4,0),(2,5)` has area `10`; the interior point `(2,2)` cannot help).

## Background

Two structural facts frame the problem:

- **The optimum lives on the convex hull.** For any maximum-area triangle, each of its three vertices
  must be a vertex of the convex hull of the point set: if a vertex were strictly inside the hull, it
  could be pushed outward to a hull vertex without ever decreasing the area. So we may first reduce the
  `n` points to their convex hull (Andrew's monotone chain, `O(n log n)`) and then work only on the
  hull polygon.
- **On a convex polygon the "obvious" linear algorithm is incorrect.** The classical idea is to walk a
  root vertex around the hull while two "caliper" pointers chase it, in the hope of an `O(n)` sweep.
  That algorithm (Dobkin–Snyder) was shown to *fail* on a 9-gon — it can miss the global optimum — so
  it cannot be used as the judge's reference. The correct, well-established replacement fixes one apex
  at a time and uses a monotone two-pointer *reset per apex*, giving `O(h^2)` on a hull of size `h`.

The non-obvious content is therefore (a) recognising the hull reduction, and (b) recognising that the
seductive single-sweep `O(n)` calipers is wrong and must be replaced by the per-apex `O(h^2)` scan.

## Evaluation settings

Judged on hidden tests covering: tiny degenerate inputs (`n = 0, 1, 2`), all-collinear sets (answer
`0`), sets with duplicate points, sets with interior points that must be ignored, points on a circle
(hull equals the whole set, the `O(h^2)` worst case), and extreme coordinates near `10^9` (so twice the
area approaches `4 * 10^18` and must stay inside signed 64-bit range).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

struct P { ll x, y; };

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<P> pts(n);
    for (int i = 0; i < n; i++) cin >> pts[i].x >> pts[i].y;

    // TODO: reduce to the convex hull, then find the maximum-area triangle
    //       on the hull; output twice that area (0 if none exists).
    ll answer = 0;

    cout << answer << "\n";
    return 0;
}
```
