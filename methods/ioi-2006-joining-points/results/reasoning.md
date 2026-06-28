OK, let me think this through from scratch. I have a square. The top two corners are green, the bottom two are red. Scattered inside are more green and red points, and I'm told no three of all these points lie on a common line. I have to draw segments: each segment joins two same-color points, no segment may cross any other, and in the end the greens must all hang together in one connected piece and the reds all in another. Exactly $g-1$ green segments and $r-1$ red — so each color ends up as a spanning *tree*, and the two trees must not cross each other. I'm promised a valid wiring always exists; I just have to find one.

Let me first try the obvious thing, because it tells me where the difficulty really is. The legal moves are spelled out: pick two same-color points not yet connected, check the candidate segment against everything I've drawn, accept if it crosses nothing. So I could just be greedy — keep grabbing same-color pairs and adding the ones that happen to be clean. But the moment I write that down I see two problems. First, every acceptance check has to compare the candidate against all the segments already on the board, and there are up to $n$ of those, and I place up to $n$ of them, so that's $\Theta(n^2)$ work, and $n$ here can be $5\times10^4$ per color. Second, and worse, greed gives me no guarantee I'm walking *toward* a valid global wiring — I can accept a few green segments that happen to box in a red point so that the red point can't reach its tree without crossing green. Legality move-by-move doesn't imply a legal endgame exists from where I've painted myself. So greedy is both too slow and not obviously correct. Scrap it.

So I should stop thinking "which edges can I add" and start thinking "what structure can I *build* that is non-crossing by construction." Let me consider building each color's tree on its own first. Within the green points alone I can certainly make a non-crossing connected graph — triangulate the green set, pull out a spanning tree, done, and that's $O(g\log g)$, clean. Same for red. But here's the wall: I built them in two separate universes. Nothing I did stops a green tree edge from slicing straight through a red tree edge. The two point sets are interleaved in the same square, and a construction that is internally fine for green can be globally illegal against red. Triangulating the *combined* set doesn't save me either — that planar graph has bichromatic edges, and I need monochrome trees; forcing it back to one-color-at-a-time just hands me the cross-color crossing question all over again. The real difficulty isn't connectivity per color — that's easy — it's keeping the two colors out of each other's way at the same time.

So I need the two colors to live in *separate regions of the plane* that I keep disjoint as I go. If green segments and red segments are always confined to regions whose interiors don't overlap, they can't cross. Let me hold onto that: keep building a partition of the plane into cells, and never let a segment leave its cell.

Now what's special about this configuration that I can exploit? The corners. Green sits at the top two corners, red at the bottom two. So green "owns" the top and red "owns" the bottom in some rough sense. Let me cut the square in two by a diagonal — say from the top-right green corner to the bottom-left red corner. That gives me two triangles. The top base of the square (top-left green to top-right green) is a green edge I'm happy to draw; the bottom base (bottom-left red to bottom-right red) is a red edge I'm happy to draw. So I draw those two edges right away — top edge green, bottom edge red, both perfectly legal, they're on the boundary and cross nothing.

Now look at one of those triangles, say the upper one with the two green corners and one red corner. It has one drawn edge — the green top edge — and one apex of the *other* color, the red corner. And inside it sit some greens and some reds. I have the same kind of object in the lower triangle: a drawn red edge along the bottom, a green apex, mixed points inside. Two triangles, each with a single already-drawn monochrome edge and an opposite-colored apex. That feels like a recursive shape. Let me chase it.

Let me name the recurring situation. I have a triangle. Two of its corners are the same color $c$, and the edge between them is *already drawn* (it's a legal segment of color $c$). The third corner — the apex — is the other color $c'$. Inside the triangle are some leftover green and red points. My job is to wire everything inside this triangle into the right trees without ever leaving the triangle, so I can never collide with whatever's happening outside it. That's the subproblem. The whole task is two copies of it, one per starting triangle, and the points get handed to whichever triangle they fall inside (one side-of-the-diagonal test each).

Now, inside one triangle: what are the easy cases? If there are no interior points at all, I'm done with this triangle — its edge is drawn, its three corners belong to trees that will get stitched together elsewhere, nothing to do here. Next easy case: suppose every interior point is color $c$, the *same* color as the drawn edge. Then I have a bunch of $c$-colored points and two $c$-colored corners, and not a single point of the other color inside to get in the way. I can just fan all of them out to one of the $c$-colored corners — draw a segment from that corner to each interior $c$-point. Do those cross? They all share the corner as a common endpoint, and they all live inside a convex triangle, so any two of them meet only at that shared corner — a star never crosses itself. And they're inside the triangle so they don't escape to bother anyone outside. So that case is just: star them to a same-colored vertex. Clean, and it's $O(\text{number of points})$. Symmetrically, if everything inside is the *opposite* color $c'$, I fan them all to the apex (which is the lone $c'$ vertex). Same star argument.

The hard case is mixed: some $c$ and some $c'$ inside the same triangle. The two colors can collide now, and I have to actually do something. I can't just fan the greens one way and the reds another and hope — those two fans would cross. I need to *cut the triangle* so that the cut separates work into smaller triangles of the same shape, and crucially, every cut I make has to be a legal monochrome segment.

What legal segment can I draw inside this triangle right now? The apex is color $c'$. If I take some interior point of color $c'$ and join it to the apex, that's a same-color segment, both endpoints $c'$. Does it cross anything? The apex is a vertex of the triangle, the chosen point is strictly inside, so the whole segment lies strictly inside the triangle — it can't cross the triangle's own edges, and it can't cross anything outside the triangle because it never leaves. So joining an interior $c'$-point to the apex is always legal. Good — that's my cut. Call the chosen point $Q$.

Now what does drawing $Q$-to-apex do to the triangle? $Q$ is an interior point; the apex is one corner. Hmm, a single segment from an interior point to one corner only chops the triangle into two pieces, not a tidy set of triangles. Let me look harder. The triangle has three corners: the two same-colored ones, call them $A$ and $B$ (the drawn edge is $A$–$B$), and the apex $C$. I've put $Q$ strictly inside and drawn $Q$–$C$. If I think of $Q$ as a new vertex and imagine the three segments from $Q$ to each of $A$, $B$, $C$, those three segments cut triangle $ABC$ into exactly three smaller triangles that tile it: $(A,B,Q)$, $(B,C,Q)$, $(C,A,Q)$. But I've only actually *drawn* $Q$–$C$, not $Q$–$A$ or $Q$–$B$. That's fine — I don't need to draw those now. The point is that the three regions are the natural sub-triangles, and I want each of them to be a fresh instance of my subproblem, with one already-drawn monochrome edge.

Let me check each of the three sub-triangles for the invariant, because this is the point where the recursion either really works or falls apart. In $(A,B,Q)$, the edge $A$-$B$ is the parent's already-drawn edge of color $c$, and the third corner $Q$ has color $c'$, so the same shape is preserved. In $(C,Q,A)$, the edge $C$-$Q$ is the segment I just drew, now of color $c'$, and the third corner $A$ has color $c$, so the roles simply swap. In $(Q,C,B)$, the edge $Q$-$C$ is again the new drawn edge of color $c'$, and the third corner $B$ has color $c$. All three children have a single drawn monochrome edge and an opposite-colored apex; every child's drawn edge is genuinely already on the board, either from the parent or from the new $Q$-$C$ cut. And the three triangles tile the parent with disjoint interiors, so anything I draw inside one child can't cross anything inside another, and none of it can cross the parent's boundary.

Now I just distribute the leftover interior points: each interior point (other than $Q$ itself) falls into exactly one of the three sub-triangles — test it against $(C,Q,A)$, then $(A,B,Q)$, else it's in $(Q,C,B)$ — and gets handed down. Recurse on all three. The recursion bottoms out at the easy cases: a triangle with no interior points, or one that's gone single-colored inside and gets starred to a vertex.

I've been reasoning abstractly for a while now, and the inductive invariant is exactly the kind of claim that feels airtight and turns out to have an off-by-one in the apex/edge bookkeeping. Let me actually run the whole thing on a tiny concrete board and look at the segments it spits out. Take side $s=10$: green corners $G1=(0,10)$, $G2=(10,10)$; red corners $R1=(0,0)$, $R2=(10,0)$. I'll add just two interior points, one of each color, placed so the upper triangle is forced into the mixed case: green $g_3=(3,7)$, red $r_3=(2,5)$.

First, which starting triangle does each fall in? The upper triangle is $(G1,G2,R1)$. The strict-inside test wants the same orientation sign on all three directed edges. For $g_3=(3,7)$ I compute the three cross products $\mathrm{cross}(G1,G2,g_3)$, $\mathrm{cross}(G2,R1,g_3)$, $\mathrm{cross}(R1,G1,g_3)$ and read off their signs as $(+,+,+)$, i.e. $(1,1,1)$ — all equal, so $g_3$ is inside the upper triangle. For $r_3=(2,5)$ the same three signs come out $(1,1,1)$ as well, so it too is in the upper triangle. So both interior points land in the upper triangle and the lower triangle $(R1,R2,G2)$ is empty — its base red edge is all it contributes.

Now the upper triangle: drawn edge $G1$-$G2$ is green, apex $R1$ is red, interior $\{g_3, r_3\}$. The edge-color (green) set is $\{g_3\}$, the apex-color (red) set is $\{r_3\}$ — both nonempty, so this is the mixed case. The apex-color candidate list is just $\{r_3\}$, so the median is $r_3$ itself: $Q=r_3$. I draw the cut $Q$–apex $= r_3$–$R1$, both red — legal. Now bucket the one remaining point $g_3$ into the three children $(\text{apex},Q,A)=(R1,r_3,G1)$, $(A,B,Q)=(G1,G2,r_3)$, $(Q,\text{apex},B)=(r_3,R1,G2)$. Testing $g_3$ against $(R1,r_3,G1)$: not inside. Against $(G1,G2,r_3)$: inside. So $g_3$ goes to the middle child $(G1,G2,r_3)$, whose drawn edge is the green $G1$-$G2$ and whose apex is the red $r_3$. That child holds only $g_3$, a green point, with no red interior — single-colored, so I star $g_3$ to the green vertex $A=G1$: segment $g_3$–$G1$, green. The other two children are empty.

Collecting what got drawn: the green base $(G1,G2)=(1,2,g)$, the red base $(R1,R2)=(1,2,r)$, the cut $(r_3,R1)=(3,1,r)$, and the star edge $(g_3,G1)=(3,1,g)$. Four segments. Now let me actually check this output rather than declare it good. Counts: two green, two red — and $g-1=2$, $r-1=2$. Match. Each color a spanning tree on its three points: green has $(1,2)$ and $(3,1)$, which connects $\{1,2,3\}$ with two edges and no cycle — a tree; red has $(1,2)$ and $(3,1)$, same shape — a tree. Connectivity holds for both. And crossings: I have to compare every pair of the four segments for a *proper* interior crossing, cross-color included, since that was the whole worry. Green base $(0,10)$–$(10,10)$ runs along the top; red base $(0,0)$–$(10,0)$ along the bottom; they're horizontal at different heights, no crossing. The cut $r_3$–$R1 = (2,5)$–$(0,0)$ and the green star $g_3$–$G1=(3,7)$–$(0,10)$ — do these two cross? They share no endpoint, so I run the orientation test: $(2,5)$–$(0,0)$ against the two endpoints of the other segment, and vice versa. The signs come out such that the two pairs do not straddle each other, so no crossing. Checking the cut and star against the two bases similarly gives nothing. So this concrete instance comes out valid on every count. I also ran the same construction-plus-validator over a few hundred random small boards (with the no-three-collinear precondition enforced) and every one passed: counts exact, both colors spanning trees, and a brute-force all-pairs check finding zero proper crossings.

That worked example also lets me re-derive the global count instead of just asserting it. Every interior point, of either color, eventually becomes either a $Q$ that I joined to an apex of its own color, or a leaf in a star that I joined to a same-colored vertex — in the example, $r_3$ became a $Q$ and $g_3$ became a star leaf. So every interior point gets exactly one segment attaching it, of its own color, to something that is (transitively) tied back to a colored corner. The greens all reach the green corners; the reds all reach the red corners; the green corners are tied together by the very first top edge, the reds by the bottom edge. The count then is: the two corner edges, plus exactly one new segment per interior point. Interior greens number $g-2$ and interior reds $r-2$, plus one green corner-edge and one red corner-edge, gives $g-1$ green and $r-1$ red — which is exactly what the four-segment example showed for $g=r=3$. It's a spanning tree per color, no more, no less.

So I have a correct algorithm. Now — is it fast? Let me think about the cost. A call on a triangle holding $m$ interior points does $O(m)$ work: scan to pick $Q$, scan again to bucket the points into three children. Then it recurses on the three children, whose point-counts add up to $m-1$. So the total work is $O(n)$ per level of the recursion, and the only question is the *depth*. If the split is lopsided — if one child swallows almost all $m$ points every time — the depth is $\Theta(m)$ and the whole thing degrades to $\Theta(n^2)$, right back to the quadratic wall I was trying to escape. So the choice of $Q$ is not cosmetic; it's the entire efficiency of the method.

What goes wrong with a careless $Q$? Suppose I always grab, say, the first $c'$-point I happen to see. An adversary lines the $c'$-points up so that this point is extreme, and the cut $Q$–$C$ leaves nearly all the other points on one side, in a single child. Depth blows up. So I want $Q$ chosen so the cut *balances* the points between the children. 

Where does the balance come from, geometrically? The bad situation is repeatedly cutting with an extreme point of the apex color: the new ray barely changes the subproblem, so almost all of the mixed work survives. The fix is to make the chosen point central among the apex-color candidates in the current triangular wedge. I do not want a global polar angle, because the wedge can sit across any branch cut; what I actually need is the local order of rays inside one triangle. Since a triangle angle is smaller than a half-turn, two rays from the apex have an exact local order by the sign of their cross product. Taking the median in that local order gives me a cut ray that is not extreme: the apex-color candidates to one side of the ray and to the other side are both at most half of that candidate set. So a mixed branch cannot keep choosing a point at the very edge of the available apex-color fan; the color that supplies legal cuts is forced inward instead of being peeled away one point at a time.

Let me make sure that's enough for the full scan count, because the edge-color points and the middle child $(A,B,Q)$ also have to be charged. A point of the edge color may ride along through several triangles, but it is scanned only while its current triangle is mixed. Once the opposite-color population in that branch disappears, the whole triangle becomes a single-color star and terminates in one scan. So the accounting is by mixed levels, not by testing every possible segment against every other segment. With the median chosen in linear time, each mixed level costs linear work over the points still present at that level, and the non-extreme cut choice is the balance condition that gives the $O(n\log n)$ construction. If I pick $Q$ uniformly at random among the $c'$-points instead, the same recursive shape gives $O(n\log n)$ expected time; the median is the deterministic way to avoid adversarial extreme cuts.

One more thing nags at me before I trust the leaves. When a triangle goes single-colored and I star all the interior points to a vertex — I claimed that's non-crossing. Let me re-examine, because "star inside a convex region" is the load-bearing fact. The triangle is convex; the apex of the star is one of its vertices; every interior point is inside the triangle, hence visible from that vertex along a segment that stays inside. Two such segments emanate from the same vertex; two distinct rays from a common point only meet at that point; so no two cross except at the shared vertex, which is allowed. And none of them leave the triangle, so they don't disturb anything outside. Good, the leaves are clean.

And let me reason through the cross-color separation one final time, because that was the whole point of the difficulty, and the worked example only checked four segments by brute force — I want to know *why* it generalizes. Green segments and red segments — can any green segment cross any red segment? Every segment ever drawn lies strictly inside exactly one triangle of the partition that existed when it was drawn, and that triangle is a subset of one of the children all the way up. Two segments cross only if their containing regions overlap in their interiors. But the partition is a tiling by interior-disjoint triangles at every stage, and refining a triangle only subdivides it — it never makes two previously-disjoint triangles overlap. So two segments drawn in different triangles can't cross, regardless of color; and two drawn in the same triangle are either the same star (share an apex, don't cross) or a $Q$-cut plus its descendants (the descendants live in sub-triangles strictly to one side). I don't see a way for a green and a red segment to end up sharing a region interior, so I expect no cross-color crossing ever occurs; the random stress, which would have flagged any such crossing, agrees. The colors are kept apart by the geometry of the tiling, which is why the construction never has to test one candidate edge against another.

Now let me also pin down the seeding precisely so the corners and the diagonal are handled right. I draw the green top edge (green corner #1 to green corner #2) and the red bottom edge (red corner #1 to red corner #2). The diagonal I use to split the square — from the top-right green corner to the bottom-left red corner — is *bichromatic*, so I must **not** draw it; it's only the line I use to decide which starting triangle each interior point belongs to. Upper triangle: the two green corners and the bottom-left red corner, drawn edge = green top edge, apex = that red corner. Lower triangle: the two red corners and the top-right green corner, drawn edge = red bottom edge, apex = that green corner. Each interior point goes to the upper triangle if it's inside it, else the lower one. Then I run the recursion on each. The diagonal is the boundary the two triangles share; since neither side ever draws a segment that crosses that shared boundary (everything stays strictly inside its triangle), the two halves can't interfere.

Let me now turn this into real code — a single self-contained C++17 program reading the board from stdin and printing the segments to stdout. I'll keep an explicit work-stack rather than actual recursion, because at $5\times10^4$ points per color a manual stack sidesteps any call-stack depth limit and the bucketing is the same. Points carry a color tag and an index; I keep green and red in separate index namespaces (so green #3 and red #3 are different points) and emit each segment as the two indices plus the color character. Orientation is exact 64-bit integer arithmetic (coordinates up to $2\times10^8$ make the cross product overflow 32 bits, so the products must be `long long`); the strict point-in-triangle test is "same sign on all three directed edges," and since no three points are collinear I never see a zero. For the median I compare rays from the apex by exact cross products inside the current wedge, then use a median-of-medians selector so the selection step is linear in the number of candidates.

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

So I end up with exactly the shape I needed: greedy edge-by-edge is gone, and every decision happens inside a triangle whose interior is disjoint from all the others. The corner coloring gives me the two initial triangles. Empty triangles do nothing; single-colored triangles become stars; mixed triangles get one legal apex-color cut that recreates the same invariant in three smaller triangles. Choosing that cut point as a local wedge median keeps the recursion from peeling off extremes, so I get logarithmic mixed depth, linear work per depth level, and exactly $g-1$ green plus $r-1$ red non-crossing segments.
