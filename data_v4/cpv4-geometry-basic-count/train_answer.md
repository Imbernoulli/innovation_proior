**Problem.** Given `n` integer points (`0 <= n <= 2*10^5`, coordinates up to `10^9` in magnitude), count the distinct **axis-aligned right triangles** whose three corners are points of the set — one right angle whose legs are a vertical and a horizontal segment. A triangle is its set of three corners; coincident input points are one location, and a triangle needs three distinct positions. Read `n` and the points from stdin, print the count.

**Key idea — per-anchor counting.** Every axis-aligned right triangle has a *unique* right-angle vertex `P`: from `P` one leg runs vertically (a corner sharing `P`'s x) and the other horizontally (a corner sharing `P`'s y); the other two corners are not right angles. So each triangle is owned by exactly one of its corners. For each distinct point `P=(px,py)`, the triangles anchored at `P` are
`(#distinct points in column px other than P) * (#distinct points in row py other than P)`,
the two leg choices being independent. Summing this over all distinct points counts every triangle exactly once — no division, no correction. With hash/ordered maps for column and row populations it is `O(n log n)`.

**Why the obvious per-pair count is wrong.** "Count vertical segments `V` (pairs sharing a column) and horizontal segments `H`, then combine them" over-counts: a vertical and a horizontal segment need not meet at a shared vertex, so most `(V,H)` pairs are not triangles. On the sample `(0,0),(0,2),(0,3),(2,0),(3,0)` it gives `C(3,2)*C(3,2)=9`, but the answer is `4`. The honest quantity is "legs that share their right-angle endpoint", which is exactly the per-anchor product.

**Pitfalls.**
1. *Double-count via per-pair / wrong anchor.* Attribute each triangle to its unique right-angle vertex and sum a *product* per anchor; do not glue independent pairs. The unique-vertex argument is what guarantees no triangle is counted twice with no division.
2. *Duplicate points.* Collapse exact duplicates to one location *before* counting, and sum over distinct points. Counting over raw input both repeats an anchor's term and inflates column/row populations (a "leg" drawn to `P` itself). Trace `(0,0),(0,0),(5,0)`: counting raw gives `4` phantom triangles for two distinct positions; dedup-then-count gives the correct `0`.
3. *Exclude-self off-by-one.* Subtract `1` from *both* the column count and the row count (`P` is not its own leg endpoint). Dropping it on either factor double-counts or invents triangles: one isolated point would yield `col*row = 1` instead of `0`.
4. *Overflow.* A `447 x 447` grid yields `447^2 * 446^2 ~ 3.97*10^10` triangles, past 32-bit. Use `long long`. (Coordinates are only map keys, never multiplied, so their magnitude is harmless.)

**Edge cases.** `n = 0` -> `0`; fewer than three distinct points -> `0` (every term has a zero factor); all points collinear (one shared row or column) -> `0`; duplicate-heavy input collapses correctly.

**Complexity.** `O(n log n)` time (ordered-map keying on coordinates), `O(n)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> xs(n), ys(n);
    // dedup identical points: a multiset of coincident points is one location
    set<pair<long long,long long>> seen;
    vector<pair<long long,long long>> pts;
    for (int i = 0; i < n; i++) {
        long long x, y;
        cin >> x >> y;
        if (seen.insert({x, y}).second) pts.push_back({x, y});
    }
    int m = (int)pts.size();

    // colCount[x] = number of DISTINCT points with that x; rowCount[y] similarly
    map<long long, long long> colCount, rowCount;
    for (auto &p : pts) {
        colCount[p.first]++;
        rowCount[p.second]++;
    }

    // For each point P, right triangles with the right angle AT P and legs
    // parallel to the axes = (#points sharing P's column, other than P)
    //                       * (#points sharing P's row, other than P).
    // Each such triangle has a unique right-angle vertex, so summing over P
    // counts every triangle exactly once -- no division, no double counting.
    long long answer = 0;
    for (auto &p : pts) {
        long long up = colCount[p.first] - 1;  // exclude P itself
        long long side = rowCount[p.second] - 1;
        answer += up * side;
    }

    cout << answer << "\n";
    return 0;
}
```
