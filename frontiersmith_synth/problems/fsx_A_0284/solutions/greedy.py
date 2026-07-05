# TIER: greedy
# Multi-restart uniform random sampling inside the unit square: draw many
# random layouts, keep the one with the largest minimum triangle area, and
# floor it at the ring baseline so it never does worse than trivial.
import sys, math, random

RING_R = 0.20
CX, CY = 0.5, 0.5


def tri_area(p, q, r):
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


def min_tri(pts):
    n = len(pts)
    best = float("inf")
    for a in range(n):
        pa = pts[a]
        for b in range(a + 1, n):
            pb = pts[b]
            for c in range(b + 1, n):
                ar = tri_area(pa, pb, pts[c])
                if ar < best:
                    best = ar
    return best


def ring(N):
    return [(CX + RING_R * math.cos(2 * math.pi * k / N),
             CY + RING_R * math.sin(2 * math.pi * k / N)) for k in range(N)]


def main():
    N = int(sys.stdin.read().split()[0])
    rng = random.Random(9001 + N)
    best_pts = ring(N)
    best_val = min_tri(best_pts)
    restarts = 2500
    for _ in range(restarts):
        pts = [(rng.random(), rng.random()) for _ in range(N)]
        v = min_tri(pts)
        if v > best_val:
            best_val = v
            best_pts = pts
    out = ["%.12f %.12f" % (x, y) for (x, y) in best_pts]
    sys.stdout.write("\n".join(out) + "\n")


main()
