# Recursive triangle splitting

## Problem

A square has its two top corners green and its two bottom corners red. Inside sit more green and red points; in total $g$ green and $r$ red, with no three points collinear. Join same-color points by straight segments so that no two segments cross (except at shared endpoints), all greens form one connected component using exactly $g-1$ segments, and all reds form a separate component using $r-1$ segments. Output the segment list. Constraints: $g,r \le 5\times10^4$; coordinates up to $2\times10^8$ (orientation arithmetic needs 64-bit integers).

## Key idea

Never test edges against each other. Instead keep a partition of the square into **interior-disjoint triangles** and force every drawn segment to stay strictly inside its triangle — then no two segments can ever cross, *including across colors*, which is the only hard part of the problem.

The recurring object is a triangle with **one already-drawn monochrome edge** of color $c$ and an **apex of the opposite color** $c'$, holding some interior green/red points. Three cases:

- **No interior points** → nothing to do.
- **All interior points one color** → connect each to a vertex of that color (a star inside a convex triangle is non-crossing).
- **Mixed** → pick an interior point $Q$ of the **apex color** $c'$ and draw $Q$–apex (legal: same color, lies strictly inside the triangle). This cut subdivides the triangle into three sub-triangles that tile it — $(\text{apex},Q,A)$, $(A,B,Q)$, $(Q,\text{apex},B)$ — each again having one drawn monochrome edge and an opposite-colored apex. Recurse.

**Seeding.** Draw the green top edge and the red bottom edge. The diagonal from the top-right green corner to the bottom-left red corner is bichromatic, so it is *not* drawn — it only splits the points into the upper triangle (two green corners + bottom-left red corner) and the lower triangle (two red corners + top-right green corner).

**Why it is correct.** Every segment is monochrome and lies strictly inside one triangle of an interior-disjoint tiling, so the drawing is plane and the two colors never cross. Each interior point gets exactly one attaching segment of its own color back toward a same-colored corner; the corners of each color are tied together by the base edge. So each color is one non-crossing spanning tree, with exactly $g-1$ green and $r-1$ red segments.

**Why $O(n\log n)$: the median.** A call on $m$ interior points costs $O(m)$ and recurses on children summing to $m-1$ points, so cost is $O(n)$ per mixed recursion layer and the depth decides everything. Choose $Q$ as the median of the apex-color points in the current triangular wedge. The code compares rays from the apex by exact cross products, which is a total local order because a triangle angle is smaller than a half-turn. The median cut prevents repeatedly choosing an extreme apex-color point: both sides of the cut inherit at most half of the current apex-color candidates, and any branch that loses one color terminates immediately as a star. With linear-time median selection this gives the $O(n\log n)$ construction. (A uniformly random apex-color point gives $O(n\log n)$ expected time; the median gives the deterministic balance.)

## Geometric primitives

- `cross(A,B,C) = (A.x-B.x)(C.y-B.y) - (C.x-B.x)(A.y-B.y)` — twice the signed area; sign gives orientation; exact in 64-bit integers.
- Strict point-in-triangle: $X$ has the same orientation sign against all three directed edges of the triangle. No three points collinear ⇒ the sign is never zero.

## Code

A single-file C++17 program. It reads `g`, the `g` green coordinates, then `r`, the `r` red coordinates from stdin, and writes `(g-1)+(r-1)` lines `i j c` (`c`=`g`/`r`) to stdout. Orientation arithmetic is exact in 64-bit integers; the recursion is an explicit work-stack to avoid any depth limit at this scale.

```cpp
// Reads: g, then g green coords; then r, then r red coords (integers, stdin).
// Writes: (g-1)+(r-1) lines "i j c" (c='g'/'r') -- a non-crossing green spanning
// tree and a non-crossing red spanning tree, via recursive triangle splitting.
#include <cstdio>
#include <vector>
#include <algorithm>
using namespace std;

struct Pt { long long x, y; int color; int ind; }; // color 0=green, 1=red
static vector<Pt> P; // 1-indexed; greens 1..g, reds at g+1..g+r (ind is per-color)

// twice the signed area of (A,B,C); exact in 64-bit ints; sign gives orientation
static inline long long cross(const Pt& A, const Pt& B, const Pt& C) {
    return (A.x - B.x) * (C.y - B.y) - (C.x - B.x) * (A.y - B.y);
}
static inline int sgn(long long v) { return v > 0 ? 1 : (v < 0 ? -1 : 0); }

// strict interior: same orientation sign on all three directed edges
static inline bool inside(const Pt& A, const Pt& B, const Pt& C, const Pt& X) {
    int s1 = sgn(cross(A, B, X));
    int s2 = sgn(cross(B, C, X));
    int s3 = sgn(cross(C, A, X));
    return s1 == s2 && s2 == s3;
}

// median (lower middle, index n/2) of apex-color point ids ordered by ray angle
// from the apex, via median-of-medians so selection is linear in #candidates.
static long long apexX, apexY; // apex coords for the wedge comparator
// true iff ray apex->u precedes ray apex->v locally (wedge angle < half-turn)
static inline bool angleLess(int u, int v) {
    return (P[u].x - apexX) * (P[v].y - apexY) - (P[v].x - apexX) * (P[u].y - apexY) > 0;
}
static void smallSort(vector<int>& a, int lo, int hi) { // insertion sort [lo,hi)
    for (int i = lo + 1; i < hi; ++i) {
        int x = a[i], j = i;
        while (j > lo && angleLess(x, a[j - 1])) { a[j] = a[j - 1]; --j; }
        a[j] = x;
    }
}
static int selectK(vector<int> a, int k) { // k-th smallest (0-indexed) under angleLess
    while (true) {
        int n = (int)a.size();
        if (n <= 32) { smallSort(a, 0, n); return a[k]; }
        vector<int> medians;
        for (int i = 0; i < n; i += 5) {
            int hi = min(i + 5, n);
            smallSort(a, i, hi);
            medians.push_back(a[i + (hi - i) / 2]);
        }
        int pivot = selectK(medians, (int)medians.size() / 2);
        vector<int> lows, equals, highs;
        for (int x : a) {
            if (x == pivot) equals.push_back(x);
            else if (angleLess(x, pivot)) lows.push_back(x);
            else if (angleLess(pivot, x)) highs.push_back(x);
            else equals.push_back(x);
        }
        if (k < (int)lows.size()) a.swap(lows);
        else if (k < (int)lows.size() + (int)equals.size()) return equals[0];
        else { k -= (int)lows.size() + (int)equals.size(); a.swap(highs); }
    }
}

// A subproblem triangle: drawn monochrome edge A-B (color = P[A].color),
// apex C of the opposite color, and the interior point ids in pts.
struct Frame { int A, B, C; vector<int> pts; };

int main() {
    int g, r;
    if (scanf("%d", &g) != 1) return 0;
    P.assign(1, Pt{}); // dummy 0
    for (int i = 1; i <= g; ++i) { Pt p; scanf("%lld %lld", &p.x, &p.y); p.color = 0; p.ind = i; P.push_back(p); }
    scanf("%d", &r);
    for (int i = 1; i <= r; ++i) { Pt p; scanf("%lld %lld", &p.x, &p.y); p.color = 1; p.ind = i; P.push_back(p); }

    int G1 = 1, G2 = 2;           // green corners (top-left #1, top-right #2)
    int R1 = g + 1, R2 = g + 2;   // red corners (bottom-left #1, bottom-right #2)

    // the two monochrome base edges -- always legal, on the boundary
    printf("1 2 g\n");
    printf("1 2 r\n");

    // diagonal G2--R1 is bichromatic: NOT drawn, only used to split the points.
    // Upper triangle (G1,G2,R1): drawn edge G1-G2 (green), apex R1.
    // Lower triangle (R1,R2,G2): drawn edge R1-R2 (red),   apex G2.
    vector<int> upper, lower;
    for (int i = 1; i <= g + r; ++i) {
        if (i == G1 || i == G2 || i == R1 || i == R2) continue;
        (inside(P[G1], P[G2], P[R1], P[i]) ? upper : lower).push_back(i);
    }

    vector<Frame> stack;
    stack.push_back({G1, G2, R1, move(upper)});
    stack.push_back({R1, R2, G2, move(lower)});

    while (!stack.empty()) {
        Frame f = move(stack.back()); stack.pop_back();
        if (f.pts.empty()) continue;        // empty triangle: nothing to do
        int A = f.A, B = f.B, C = f.C;
        int cEdge = P[A].color;             // color of the drawn edge A-B
        int cApex = P[C].color;             // opposite color
        char chEdge = cEdge ? 'r' : 'g', chApex = cApex ? 'r' : 'g';
        vector<int> same, opp;
        for (int k : f.pts) (P[k].color == cEdge ? same : opp).push_back(k);

        if (opp.empty()) {                  // all interior are edge-color:
            for (int k : same)              // star them to a same-color vertex
                printf("%d %d %c\n", P[A].ind, P[k].ind, chEdge);
            continue;
        }
        if (same.empty()) {                 // all interior are apex-color:
            for (int k : opp)               // star them to the apex
                printf("%d %d %c\n", P[C].ind, P[k].ind, chApex);
            continue;
        }

        // mixed: pick Q as the median apex-color point in the current wedge
        apexX = P[C].x; apexY = P[C].y;
        int Q = selectK(opp, (int)opp.size() / 2);

        printf("%d %d %c\n", P[Q].ind, P[C].ind, chApex); // legal cut Q--apex

        // the three children tiling (A,B,C): (C,Q,A), (A,B,Q), (Q,C,B)
        vector<int> c1, c2, c3;
        for (int k : f.pts) {
            if (k == Q) continue;
            if (inside(P[C], P[Q], P[A], P[k])) c1.push_back(k);
            else if (inside(P[A], P[B], P[Q], P[k])) c2.push_back(k);
            else c3.push_back(k);
        }
        stack.push_back({C, Q, A, move(c1)}); // drawn edge C-Q, apex A
        stack.push_back({A, B, Q, move(c2)}); // drawn edge A-B, apex Q
        stack.push_back({Q, C, B, move(c3)}); // drawn edge Q-C, apex B
    }
    return 0;
}
```

## Complexity

Time $O(n\log n)$ with $n = g + r$ (median split gives $O(\log n)$ depth, $O(n)$ work per level); space $O(n)$. Output is exactly $(g-1) + (r-1)$ segments.
