We are given a square whose two top corners are green and whose two bottom corners are red, with more green and red points scattered inside, $g$ green and $r$ red in total and no three of them collinear. We must join same-color points by straight segments so that no two segments cross except at shared endpoints, the greens form one connected component using exactly $g-1$ segments and the reds a separate one using exactly $r-1$ segments — that is, each color becomes a non-crossing spanning tree and the two trees must also avoid each other. A valid wiring is promised; we just have to produce one fast, with $g,r$ up to $5\times10^4$ and coordinates up to $2\times10^8$ so that orientation arithmetic must stay in exact 64-bit integers.

The obvious move is to read the legality rules literally and be greedy: repeatedly pick two same-color points not yet connected, test the candidate segment against everything already drawn, and accept it if it crosses nothing. That fails on two counts. Each acceptance test compares the candidate against all the up-to-$n$ segments already placed, and we place up to $n$ of them, so the work is $\Theta(n^2)$ — quadratic at $5\times10^4$ points per color. Worse, legality move-by-move does not imply a legal endgame exists: a few innocent green segments can box in a red point so it can no longer reach its own tree without crossing green, and greed gives no signal that we have painted ourselves into that corner. Building each color's tree in its own universe does not save us either. Triangulating just the green points and pulling out a spanning tree is easy and fast, but nothing in that construction stops a green tree edge from slicing straight through a red one, because the two point sets are interleaved in the same square. Triangulating the combined set produces bichromatic edges, which are exactly what we are not allowed to draw. The real difficulty is never the per-color connectivity — that is trivial — it is keeping the two colors out of each other's way simultaneously.

I propose recursive triangle splitting. The idea that dissolves the whole crossing problem is to stop asking "which edges may I add" and instead maintain a partition of the square into interior-disjoint triangles, forcing every segment I ever draw to lie strictly inside one triangle of the current partition. Two segments can cross only if their containing regions overlap in their interiors; a tiling by interior-disjoint triangles never has that, and refining a triangle only subdivides it and never makes two previously-disjoint triangles overlap. So crossing-freeness, including across colors, becomes automatic by construction and is never checked. The recurring object I carry through the recursion is a triangle with one already-drawn monochrome edge of color $c$, joining vertices I call $A$ and $B$, and an apex $C$ of the opposite color $c'$, holding some leftover interior points. The whole task is two copies of this object, and the seeding produces them: I draw the green top edge (green corner #1 to green corner #2) and the red bottom edge (red corner #1 to red corner #2), both legal because they lie on the boundary and cross nothing. The diagonal from the top-right green corner $G_2$ to the bottom-left red corner $R_1$ is bichromatic, so I must not draw it; it serves only as the line that splits the interior points into the upper triangle $(G_1,G_2,R_1)$, whose drawn edge is green and whose apex $R_1$ is red, and the lower triangle $(R_1,R_2,G_2)$, whose drawn edge is red and whose apex $G_2$ is green. One side-of-diagonal test sends each interior point to its starting triangle.

Inside one triangle there are three cases. If it holds no interior points, there is nothing to do — its edge is drawn and its corners get stitched to their trees elsewhere. If every interior point is the same color as the drawn edge, I fan all of them out to one $c$-colored corner; if every interior point is the apex color, I fan them all to the apex. A star inside a convex triangle is the load-bearing fact here: the apex of the star is a vertex of the triangle, every interior point is visible from it along a segment that stays inside, and two distinct rays from a common point meet only at that point, so the star never crosses itself and never leaves the triangle. The hard case is mixed, both colors present inside, because now the two colors can collide and I cannot simply fan greens one way and reds another. I need a cut that is itself a legal monochrome segment and that splits the work into smaller copies of the same object. The apex $C$ has color $c'$, so if I take any interior point $Q$ of color $c'$ and draw the segment $Q$–$C$, it is same-color and lies strictly inside the triangle (one endpoint a corner, the other strictly interior), hence legal against everything inside and outside. That segment is my cut. Treating $Q$ as a new vertex, the three segments from $Q$ to $A$, $B$, $C$ would tile the triangle $ABC$ into $(A,B,Q)$, $(B,C,Q)$, $(C,A,Q)$; I only actually draw $Q$–$C$, but the three regions are the natural children. Checking the invariant on each is what makes the recursion close: in $(A,B,Q)$ the drawn edge is the parent's $A$–$B$ of color $c$ and the third corner $Q$ is color $c'$; in $(C,Q,A)$ the drawn edge is the new $C$–$Q$ of color $c'$ and the apex $A$ is color $c$, roles swapped; in $(Q,C,B)$ the drawn edge is again $Q$–$C$ of color $c'$ and the apex $B$ is color $c$. Each child has exactly one already-drawn monochrome edge and an opposite-colored apex, so I recurse. Distributing the leftover points is one point-in-triangle test against $(C,Q,A)$, then $(A,B,Q)$, else $(Q,C,B)$.

This produces exactly the trees and counts required. Every interior point becomes either a $Q$ joined to an apex of its own color, or a leaf in a star joined to a same-colored vertex — in either case exactly one attaching segment of its own color, tying it transitively back to a colored corner. The green corners are bound together by the top edge, the red corners by the bottom edge, so green is one connected non-crossing tree and red another. The segment count is the two corner edges plus one new segment per interior point; with $g-2$ interior greens and $r-2$ interior reds that gives $g-1$ green and $r-1$ red segments, a spanning tree per color with nothing to spare.

What makes the method efficient — and the choice of $Q$ is the entire efficiency, not a cosmetic detail — is selecting $Q$ as the median apex-color point in the current triangular wedge. A call on $m$ interior points does $O(m)$ work (one scan to pick $Q$, one to bucket the rest into three children), and the children's point-counts sum to $m-1$, so the cost is $O(n)$ per recursion level and the depth decides everything. A careless choice, such as always grabbing the first apex-color point seen, lets an adversary place that point at the extreme of the apex-color fan so the cut barely changes the subproblem and nearly all the mixed work survives in one child; the depth then degrades to $\Theta(m)$ and we are back at the quadratic wall. The balance comes from making the chosen ray non-extreme. I do not use a global polar angle, because the wedge can straddle any branch cut; what I need is the local order of rays from the apex inside this one triangle, and because a triangle angle is smaller than a half-turn, two rays have an exact local order given by the sign of their cross product. Taking the median in that order forces both sides of the cut to inherit at most half of the apex-color candidates, so a mixed branch cannot keep peeling off extremes; the moment a branch loses one color entirely it terminates as a single star. With a linear-time selection — a median-of-medians selector using the wedge cross-product comparator — each mixed level costs linear work and the depth is $O(\log n)$, giving an $O(n\log n)$ construction in $n=g+r$ with $O(n)$ space. (A uniformly random apex-color point yields the same bound in expectation; the median is the deterministic way to defeat adversarial extreme cuts.) I keep an explicit work-stack rather than real recursion to sidestep the interpreter recursion limit at this scale, tag points with color and index in separate per-color namespaces, and do all orientation work with exact integers, where the no-three-collinear guarantee means the strict same-sign-on-all-three-edges point-in-triangle test never sees a zero.

```python
import sys

def cross(ax, ay, bx, by, cx, cy):
    # twice the signed area of (A,B,C); exact in 64-bit ints
    return (ax - bx) * (cy - by) - (cx - bx) * (ay - by)

def sgn(v):
    return 1 if v > 0 else (-1 if v < 0 else 0)

def inside(A, B, C, X, P):
    # strict interior: same orientation sign on all three directed edges
    ax, ay = P[A]; bx, by = P[B]; cx, cy = P[C]; xx, xy = P[X]
    s1 = sgn(cross(ax, ay, bx, by, xx, xy))
    s2 = sgn(cross(bx, by, cx, cy, xx, xy))
    s3 = sgn(cross(cx, cy, ax, ay, xx, xy))
    return s1 == s2 and s2 == s3

def solve(P, g, r):
    out = []
    G1, G2 = ('g', 1), ('g', 2)      # top-left, top-right corners (green)
    R1, R2 = ('r', 1), ('r', 2)      # bottom-left, bottom-right corners (red)

    def median_item(items, less):
        target = len(items) // 2

        def small_sort(a):
            for i in range(1, len(a)):
                x = a[i]
                j = i
                while j > 0 and less(x, a[j - 1]):
                    a[j] = a[j - 1]
                    j -= 1
                a[j] = x

        def select(a, k):
            a = list(a)
            while True:
                if len(a) <= 32:
                    small_sort(a)
                    return a[k]
                medians = []
                for i in range(0, len(a), 5):
                    group = a[i:i + 5]
                    small_sort(group)
                    medians.append(group[len(group) // 2])
                pivot = select(medians, len(medians) // 2)
                lows, equals, highs = [], [], []
                for x in a:
                    if x == pivot:
                        equals.append(x)
                    elif less(x, pivot):
                        lows.append(x)
                    elif less(pivot, x):
                        highs.append(x)
                    else:
                        equals.append(x)
                if k < len(lows):
                    a = lows
                elif k < len(lows) + len(equals):
                    return equals[0]
                else:
                    k -= len(lows) + len(equals)
                    a = highs

        return select(items, target)

    # the two monochrome base edges -- always legal, on the boundary
    out.append((1, 2, 'g'))
    out.append((1, 2, 'r'))

    # diagonal G2--R1 is bichromatic: NOT drawn, only used to split the points.
    # Upper triangle (G1,G2,R1): drawn edge G1-G2 (green), apex R1.
    # Lower triangle (R1,R2,G2): drawn edge R1-R2 (red),   apex G2.
    interior = [k for k in P if k not in (G1, G2, R1, R2)]
    upper, lower = [], []
    for k in interior:
        (upper if inside(G1, G2, R1, k, P) else lower).append(k)

    # work-stack of subproblems: (A, B, Apex, pts)
    #   edge A-B already drawn and monochrome; Apex is the opposite color
    stack = [(G1, G2, R1, upper), (R1, R2, G2, lower)]

    while stack:
        A, B, Apex, pts = stack.pop()
        if not pts:
            continue                         # empty triangle: nothing to do
        cEdge = A[0]                          # color of the drawn edge A-B
        cApex = Apex[0]                       # opposite color
        same = [k for k in pts if k[0] == cEdge]
        opp  = [k for k in pts if k[0] == cApex]

        if not opp:                           # all interior are edge-color:
            for k in same:                    # star them to a same-color vertex
                out.append((A[1], k[1], cEdge))
            continue
        if not same:                          # all interior are apex-color:
            for k in opp:                      # star them to the apex
                out.append((Apex[1], k[1], cApex))
            continue

        # mixed: pick Q as the median apex-color point in the current angular wedge
        apx, apy = P[Apex]
        def angle_less(U, V):
            ux, uy = P[U]; vx, vy = P[V]
            return cross(ux, uy, apx, apy, vx, vy) > 0

        Q = median_item(opp, angle_less)

        out.append((Q[1], Apex[1], cApex))    # the legal cut: Q--Apex, both apex-color

        # the three children tiling (A,B,Apex): (Apex,Q,A), (A,B,Q), (Q,Apex,B)
        c1, c2, c3 = [], [], []
        for k in pts:
            if k == Q:
                continue
            if inside(Apex, Q, A, k, P):
                c1.append(k)
            elif inside(A, B, Q, k, P):
                c2.append(k)
            else:
                c3.append(k)
        stack.append((Apex, Q, A, c1))        # drawn edge Apex-Q, apex A
        stack.append((A, B, Q, c2))           # drawn edge A-B,    apex Q
        stack.append((Q, Apex, B, c3))        # drawn edge Q-Apex, apex B

    return out

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    g = int(next(it)); P = {}
    for i in range(1, g + 1):
        P[('g', i)] = (int(next(it)), int(next(it)))
    r = int(next(it))
    for i in range(1, r + 1):
        P[('r', i)] = (int(next(it)), int(next(it)))
    out = solve(P, g, r)
    sys.stdout.write('\n'.join(f"{a} {b} {c}" for a, b, c in out) + '\n')

if __name__ == "__main__":
    main()
```
