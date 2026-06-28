We are given a square whose two top corners are green and whose two bottom corners are red, with more green and red points scattered inside, $g$ green and $r$ red in total and no three of them collinear. We must join same-color points by straight segments so that no two segments cross except at shared endpoints, the greens form one connected component using exactly $g-1$ segments and the reds a separate one using exactly $r-1$ segments — that is, each color becomes a non-crossing spanning tree and the two trees must also avoid each other. A valid wiring is promised; we just have to produce one fast, with $g,r$ up to $5\times10^4$ and coordinates up to $2\times10^8$ so that orientation arithmetic must stay in exact 64-bit integers.

The obvious move is to read the legality rules literally and be greedy: repeatedly pick two same-color points not yet connected, test the candidate segment against everything already drawn, and accept it if it crosses nothing. That fails on two counts. Each acceptance test compares the candidate against all the up-to-$n$ segments already placed, and we place up to $n$ of them, so the work is $\Theta(n^2)$ — quadratic at $5\times10^4$ points per color. Worse, legality move-by-move does not imply a legal endgame exists: a few innocent green segments can box in a red point so it can no longer reach its own tree without crossing green, and greed gives no signal that we have painted ourselves into that corner. Building each color's tree in its own universe does not save us either. Triangulating just the green points and pulling out a spanning tree is easy and fast, but nothing in that construction stops a green tree edge from slicing straight through a red one, because the two point sets are interleaved in the same square. Triangulating the combined set produces bichromatic edges, which are exactly what we are not allowed to draw. The real difficulty is never the per-color connectivity — that is trivial — it is keeping the two colors out of each other's way simultaneously.

I propose recursive triangle splitting. The idea that dissolves the whole crossing problem is to stop asking "which edges may I add" and instead maintain a partition of the square into interior-disjoint triangles, forcing every segment I ever draw to lie strictly inside one triangle of the current partition. Two segments can cross only if their containing regions overlap in their interiors; a tiling by interior-disjoint triangles never has that, and refining a triangle only subdivides it and never makes two previously-disjoint triangles overlap. So crossing-freeness, including across colors, becomes automatic by construction and is never checked. The recurring object I carry through the recursion is a triangle with one already-drawn monochrome edge of color $c$, joining vertices I call $A$ and $B$, and an apex $C$ of the opposite color $c'$, holding some leftover interior points. The whole task is two copies of this object, and the seeding produces them: I draw the green top edge (green corner #1 to green corner #2) and the red bottom edge (red corner #1 to red corner #2), both legal because they lie on the boundary and cross nothing. The diagonal from the top-right green corner $G_2$ to the bottom-left red corner $R_1$ is bichromatic, so I must not draw it; it serves only as the line that splits the interior points into the upper triangle $(G_1,G_2,R_1)$, whose drawn edge is green and whose apex $R_1$ is red, and the lower triangle $(R_1,R_2,G_2)$, whose drawn edge is red and whose apex $G_2$ is green. One side-of-diagonal test sends each interior point to its starting triangle.

Inside one triangle there are three cases. If it holds no interior points, there is nothing to do — its edge is drawn and its corners get stitched to their trees elsewhere. If every interior point is the same color as the drawn edge, I fan all of them out to one $c$-colored corner; if every interior point is the apex color, I fan them all to the apex. A star inside a convex triangle is the load-bearing fact here: the apex of the star is a vertex of the triangle, every interior point is visible from it along a segment that stays inside, and two distinct rays from a common point meet only at that point, so the star never crosses itself and never leaves the triangle. The hard case is mixed, both colors present inside, because now the two colors can collide and I cannot simply fan greens one way and reds another. I need a cut that is itself a legal monochrome segment and that splits the work into smaller copies of the same object. The apex $C$ has color $c'$, so if I take any interior point $Q$ of color $c'$ and draw the segment $Q$–$C$, it is same-color and lies strictly inside the triangle (one endpoint a corner, the other strictly interior), hence legal against everything inside and outside. That segment is my cut. Treating $Q$ as a new vertex, the three segments from $Q$ to $A$, $B$, $C$ would tile the triangle $ABC$ into $(A,B,Q)$, $(B,C,Q)$, $(C,A,Q)$; I only actually draw $Q$–$C$, but the three regions are the natural children. Checking the invariant on each is what makes the recursion close: in $(A,B,Q)$ the drawn edge is the parent's $A$–$B$ of color $c$ and the third corner $Q$ is color $c'$; in $(C,Q,A)$ the drawn edge is the new $C$–$Q$ of color $c'$ and the apex $A$ is color $c$, roles swapped; in $(Q,C,B)$ the drawn edge is again $Q$–$C$ of color $c'$ and the apex $B$ is color $c$. Each child has exactly one already-drawn monochrome edge and an opposite-colored apex, so I recurse. Distributing the leftover points is one point-in-triangle test against $(C,Q,A)$, then $(A,B,Q)$, else $(Q,C,B)$.

This produces exactly the trees and counts required. Every interior point becomes either a $Q$ joined to an apex of its own color, or a leaf in a star joined to a same-colored vertex — in either case exactly one attaching segment of its own color, tying it transitively back to a colored corner. The green corners are bound together by the top edge, the red corners by the bottom edge, so green is one connected non-crossing tree and red another. The segment count is the two corner edges plus one new segment per interior point; with $g-2$ interior greens and $r-2$ interior reds that gives $g-1$ green and $r-1$ red segments, a spanning tree per color with nothing to spare.

What makes the method efficient — and the choice of $Q$ is the entire efficiency, not a cosmetic detail — is selecting $Q$ as the median apex-color point in the current triangular wedge. A call on $m$ interior points does $O(m)$ work (one scan to pick $Q$, one to bucket the rest into three children), and the children's point-counts sum to $m-1$, so the cost is $O(n)$ per recursion level and the depth decides everything. A careless choice, such as always grabbing the first apex-color point seen, lets an adversary place that point at the extreme of the apex-color fan so the cut barely changes the subproblem and nearly all the mixed work survives in one child; the depth then degrades to $\Theta(m)$ and we are back at the quadratic wall. The balance comes from making the chosen ray non-extreme. I do not use a global polar angle, because the wedge can straddle any branch cut; what I need is the local order of rays from the apex inside this one triangle, and because a triangle angle is smaller than a half-turn, two rays have an exact local order given by the sign of their cross product. Taking the median in that order forces both sides of the cut to inherit at most half of the apex-color candidates, so a mixed branch cannot keep peeling off extremes; the moment a branch loses one color entirely it terminates as a single star. With a linear-time selection — a median-of-medians selector using the wedge cross-product comparator — each mixed level costs linear work and the depth is $O(\log n)$, giving an $O(n\log n)$ construction in $n=g+r$ with $O(n)$ space. (A uniformly random apex-color point yields the same bound in expectation; the median is the deterministic way to defeat adversarial extreme cuts.) I keep an explicit work-stack rather than real recursion to sidestep the interpreter recursion limit at this scale, tag points with color and index in separate per-color namespaces, and do all orientation work with exact integers, where the no-three-collinear guarantee means the strict same-sign-on-all-three-edges point-in-triangle test never sees a zero.

The deliverable is a single self-contained C++17 program. It reads `g`, the `g` green coordinates, then `r`, the `r` red coordinates from stdin, and writes the `(g-1)+(r-1)` segments as lines `i j c` (`c`=`g`/`r`) to stdout. Orientation arithmetic stays exact in 64-bit integers, and the recursion is carried by an explicit work-stack so there is no depth limit at this scale.

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
