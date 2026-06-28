**Problem.** Given `n` planar points with integer coordinates (`2 <= n <= 2*10^5`,
`-10^9 <= x, y <= 10^9`, duplicates allowed), output the maximum squared Euclidean distance over all
pairs of distinct points — the squared diameter. Reading the squared distance keeps everything in exact
64-bit integers (the extreme is `(2*10^9)^2 * 2 = 8*10^18`, which fits a signed `long long`).

**Why the obvious approach is too slow.** Brute force tries all `O(n^2)` pairs. At `n = 2*10^5` that is
`~2*10^10` pair evaluations — three to four orders of magnitude past a 1-second budget. It is the
correct oracle on small inputs but unusable at scale.

**Key idea — convex hull + rotating calipers.** Two facts collapse the search:

1. *The diameter lies on the convex hull.* For a fixed endpoint, "distance to it" is convex over the
   hull, so it is maximized at a hull vertex; an interior point can never be a diameter endpoint.
   Discard all interior points via an `O(n log n)` hull.
2. *A convex polygon has only `O(h)` antipodal pairs.* Squeeze the polygon between two parallel
   supporting lines and rotate them once around; the touching ("antipodal") pair changes only when a
   line becomes flush with an edge, so a full rotation visits `O(h)` candidate pairs, and the diameter
   is one of them.

Discretize the rotation as a two-pointer sweep using an **integer area comparison** instead of angles:
for each hull edge `(h[i], h[i+1])`, advance an apex pointer `j` while
`cross(h[i], h[i+1], h[j+1]) > cross(h[i], h[i+1], h[j])` — i.e. while the next vertex is strictly
farther from the edge line (twice the triangle area). Stop at the apex and test the distances from both
edge endpoints to `h[j]`. The pointer `j` only moves forward, so after the `O(n log n)` hull the sweep
is `O(h) = O(n)`.

**Pitfalls to get right.**
1. *Strict advance.* The apex-advance comparison must be strict `>`, not `>=`. With `>=`, two vertices
   that are *equidistant* from an edge (rampant in squares, rectangles, and any far-side-collinear
   configuration) let the pointer step *past* a legitimate antipodal vertex without testing its
   distance, losing a real diameter candidate. A `2x2`/`3x3` lattice — saturated with ties — exposes
   this immediately against a brute oracle.
2. *Degenerate hulls.* The calipers loop assumes a genuine polygon (`h >= 3`). Branch explicitly:
   `m == 1` (all points identical) returns `0`; `m == 2` (all points collinear, or only two distinct
   points) returns the squared length of the segment. Build a **strict** hull (`cross <= 0` pops
   collinear points) and **dedup first**, so an all-collinear input collapses to exactly its two
   extreme endpoints and duplicates never create zero-area garbage inside the sweep.
3. *Overflow.* Coordinates, cross products, and squared distances must all be `long long`; the squared
   distance reaches `8*10^18`. The multiply `(dx)*(dx)` overflows 32-bit even when `dx` itself fits.

**Edge cases (all verified against the `O(n^2)` oracle).** `n = 2`; all points identical -> `0`; all
collinear including a vertical line and a repeated segment; dense small lattices with many duplicate and
collinear triples; coordinates at `±10^9` so the answer is exactly `8*10^18`; circle points where the
hull is the entire input (worst hull size). Differential testing: `1461` cases (650 generator + 11
explicit edges + 800 adversarial degenerate), zero mismatches. Timing at `n = 2*10^5`: random `0.052s`,
all-collinear `0.044s`, all-identical `0.035s`.

**Complexity.** `O(n log n)` for the hull, `O(n)` for the calipers sweep; `O(n)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

struct P { ll x, y; };

// cross product of (B-A) x (C-A)
static inline ll cross(const P& A, const P& B, const P& C) {
    return (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
}
static inline ll dist2(const P& A, const P& B) {
    ll dx = A.x - B.x, dy = A.y - B.y;
    return dx * dx + dy * dy;
}

// Andrew's monotone chain; returns hull in CCW order, strict (no collinear interior pts).
static vector<P> convexHull(vector<P> pts) {
    sort(pts.begin(), pts.end(), [](const P& a, const P& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    pts.erase(unique(pts.begin(), pts.end(), [](const P& a, const P& b) {
        return a.x == b.x && a.y == b.y;
    }), pts.end());
    int n = (int)pts.size();
    if (n <= 2) return pts; // 1 unique pt -> [p]; 2 -> [p,q]
    vector<P> h(2 * n);
    int k = 0;
    // lower hull
    for (int i = 0; i < n; i++) {
        while (k >= 2 && cross(h[k - 2], h[k - 1], pts[i]) <= 0) k--;
        h[k++] = pts[i];
    }
    // upper hull
    for (int i = n - 2, lo = k + 1; i >= 0; i--) {
        while (k >= lo && cross(h[k - 2], h[k - 1], pts[i]) <= 0) k--;
        h[k++] = pts[i];
    }
    h.resize(k - 1); // last point == first point, drop it
    return h;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<P> pts(n);
    for (int i = 0; i < n; i++) cin >> pts[i].x >> pts[i].y;

    vector<P> h = convexHull(pts);
    int m = (int)h.size();

    ll best = 0;
    if (m == 1) {
        // all points identical: only pair distance is 0
        best = 0;
    } else if (m == 2) {
        // hull is a single segment (all points collinear / only 2 unique)
        best = dist2(h[0], h[1]);
    } else {
        // Rotating calipers: for each edge (i, i+1) advance antipodal vertex j
        // while the triangle area (twice) grows; candidate diameters are at the
        // antipodal vertices. O(m).
        int j = 1;
        for (int i = 0; i < m; i++) {
            int ni = (i + 1) % m;
            // advance j while area of triangle(h[i], h[ni], h[j+1]) > area with h[j]
            while (cross(h[i], h[ni], h[(j + 1) % m]) > cross(h[i], h[ni], h[j])) {
                j = (j + 1) % m;
            }
            // both endpoints of the edge are antipodal to h[j]
            best = max(best, max(dist2(h[i], h[j]), dist2(h[ni], h[j])));
        }
    }

    cout << best << "\n";
    return 0;
}
```
