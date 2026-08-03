# TIER: greedy
# Seed from the FULL inscribed-circle ring (radius 0.5) -- already larger than
# the checker's shrunk radius-0.3 baseline -- then multi-restart uniform random
# sampling inside the square, keeping the point set with the largest minimum
# triangle area. Seeded -> deterministic.
import sys, math, random

CX, CY = 0.5, 0.5


def min_area(pts):
    n = len(pts)
    best = float("inf")
    for a in range(n):
        pa = pts[a]
        for b in range(a + 1, n):
            pb = pts[b]
            bx0 = pb[0] - pa[0]; by0 = pb[1] - pa[1]
            for c in range(b + 1, n):
                pc = pts[c]
                ar = abs(bx0 * (pc[1] - pa[1]) - by0 * (pc[0] - pa[0]))
                if ar < best:
                    best = ar
    return best  # twice-area; monotone with area


def full_ring(N):
    r = 0.5
    return [(min(1.0, max(0.0, CX + r * math.cos(2 * math.pi * k / N))),
             min(1.0, max(0.0, CY + r * math.sin(2 * math.pi * k / N)))) for k in range(N)]


def main():
    t = sys.stdin.read().split()
    N = int(t[0])
    rng = random.Random(1234567)
    best_pts = full_ring(N)
    best_val = min_area(best_pts)
    R = 700
    for _ in range(R):
        pts = [(rng.random(), rng.random()) for _ in range(N)]
        v = min_area(pts)
        if v > best_val:
            best_val = v
            best_pts = pts
    out = ["%.12f %.12f" % (p[0], p[1]) for p in best_pts]
    sys.stdout.write("\n".join(out) + "\n")


main()
