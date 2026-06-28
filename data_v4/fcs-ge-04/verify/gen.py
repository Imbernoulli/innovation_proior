import sys
import random
from math import atan2, gcd

# Random small-case generator for the Pick's-theorem lattice-count problem.
# Emits a SIMPLE (non-self-intersecting), non-degenerate polygon with small
# integer coordinates so the bounding-box brute force stays fast.
# Usage: python3 gen.py <seed>

def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def seg_intersect_proper(p1, p2, p3, p4):
    # returns True if segments p1p2 and p3p4 intersect in a way that makes the
    # polygon NON-simple: i.e. they cross, or overlap, excluding the shared
    # endpoint of adjacent edges.
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    def on_seg(a, b, c):
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and \
               min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
    if d1 == 0 and on_seg(p3, p4, p1):
        return True
    if d2 == 0 and on_seg(p3, p4, p2):
        return True
    if d3 == 0 and on_seg(p1, p2, p3):
        return True
    if d4 == 0 and on_seg(p1, p2, p4):
        return True
    return False

def is_simple(pts):
    n = len(pts)
    if n < 3:
        return False
    # no zero-area total
    area2 = 0
    for i in range(n):
        j = (i + 1) % n
        area2 += pts[i][0]*pts[j][1] - pts[j][0]*pts[i][1]
    if area2 == 0:
        return False
    # no repeated vertex
    if len(set(pts)) != n:
        return False
    # check every pair of non-adjacent edges for intersection;
    # adjacent edges may only touch at their shared vertex.
    for i in range(n):
        a, b = pts[i], pts[(i+1) % n]
        for k in range(i+1, n):
            c, d = pts[k], pts[(k+1) % n]
            # skip if they share a vertex (adjacent edges)
            if i == k:
                continue
            adjacent = (k == (i+1) % n) or (i == (k+1) % n)
            if adjacent:
                # adjacent edges: forbid any intersection beyond shared endpoint
                # detect collinear overlap (spike) which makes it degenerate
                shared = set([a, b]) & set([c, d])
                others = [p for p in (a, b, c, d) if p not in shared]
                if len(shared) == 1 and len(others) == 2:
                    o = next(iter(shared))
                    if cross(o, others[0], others[1]) == 0:
                        # collinear adjacent edges -> degenerate spike
                        return False
                continue
            if seg_intersect_proper(a, b, c, d):
                return False
    return True

def gen_angle_sorted(rng, n, lim):
    # random points, sorted by angle around centroid -> usually simple
    pts = set()
    while len(pts) < n:
        pts.add((rng.randint(-lim, lim), rng.randint(-lim, lim)))
    pts = list(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    pts.sort(key=lambda p: (atan2(p[1]-cy, p[0]-cx), (p[0]-cx)**2 + (p[1]-cy)**2))
    return pts

def gen_convex_hull(rng, n, lim):
    pts = set()
    while len(pts) < n:
        pts.add((rng.randint(-lim, lim), rng.randint(-lim, lim)))
    pts = sorted(pts)
    # Andrew's monotone chain
    def build(points):
        h = []
        for p in points:
            while len(h) >= 2 and cross(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h
    lower = build(pts)
    upper = build(pts[::-1])
    hull = lower[:-1] + upper[:-1]
    return hull

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    lim = rng.choice([2, 3, 4, 5, 6, 8])
    for _ in range(400):
        mode = rng.randint(0, 2)
        if mode == 0:
            n = rng.randint(3, 9)
            pts = gen_angle_sorted(rng, n, lim)
        elif mode == 1:
            n = rng.randint(3, 12)
            pts = gen_convex_hull(rng, n, lim)
            if len(pts) < 3:
                continue
        else:
            # axis-aligned rectangle / L-shape style
            n = rng.randint(3, 8)
            pts = gen_angle_sorted(rng, n, lim)
        # remove consecutive collinear/duplicate to clean degeneracies
        cleaned = []
        m = len(pts)
        for i in range(m):
            if pts[i] == pts[(i-1) % m]:
                continue
            cleaned.append(pts[i])
        pts = cleaned
        # drop collinear middle vertices
        if len(pts) >= 3:
            tmp = []
            m = len(pts)
            for i in range(m):
                a = pts[(i-1) % m]; b = pts[i]; c = pts[(i+1) % m]
                if cross(a, b, c) != 0:
                    tmp.append(b)
            pts = tmp
        if len(pts) >= 3 and is_simple(pts):
            out = [str(len(pts))]
            for p in pts:
                out.append(f"{p[0]} {p[1]}")
            sys.stdout.write("\n".join(out) + "\n")
            return
    # fallback: a unit triangle, always valid
    sys.stdout.write("3\n0 0\n2 0\n0 2\n")

if __name__ == "__main__":
    main()
