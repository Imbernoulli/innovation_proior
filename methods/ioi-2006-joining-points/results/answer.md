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

## Complexity

Time $O(n\log n)$ with $n = g + r$ (median split gives $O(\log n)$ depth, $O(n)$ work per level); space $O(n)$. Output is exactly $(g-1) + (r-1)$ segments.
