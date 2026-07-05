# TIER: greedy
# Multi-restart uniform random sampling inside the triangle; keep the
# point set with the largest minimum-triangle-area. Seeded -> deterministic.
import sys, math, random


def read_tri():
    t = sys.stdin.read().split()
    N = int(t[0])
    A = (float(t[1]), float(t[2]))
    B = (float(t[3]), float(t[4]))
    C = (float(t[5]), float(t[6]))
    return N, A, B, C


def cross(ox, oy, px, py, qx, qy):
    return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)


def min_area(pts):
    n = len(pts)
    best = float("inf")
    for a in range(n):
        pa = pts[a]
        for b in range(a + 1, n):
            pb = pts[b]
            for c in range(b + 1, n):
                pc = pts[c]
                ar = abs(cross(pa[0], pa[1], pb[0], pb[1], pc[0], pc[1]))
                if ar < best:
                    best = ar
    return best  # twice-area; monotone with area


def sample(A, B, C, rng):
    u = rng.random(); v = rng.random()
    if u + v > 1.0:
        u, v = 1.0 - u, 1.0 - v
    x = A[0] + u * (B[0] - A[0]) + v * (C[0] - A[0])
    y = A[1] + u * (B[1] - A[1]) + v * (C[1] - A[1])
    return (x, y)


def full_incircle(A, B, C, N):
    a = math.hypot(B[0] - C[0], B[1] - C[1])
    b = math.hypot(C[0] - A[0], C[1] - A[1])
    c = math.hypot(A[0] - B[0], A[1] - B[1])
    s = 0.5 * (a + b + c)
    area = math.sqrt(max(0.0, s * (s - a) * (s - b) * (s - c)))
    r = area / s
    ix = (a * A[0] + b * B[0] + c * C[0]) / (a + b + c)
    iy = (a * A[1] + b * B[1] + c * C[1]) / (a + b + c)
    return [(ix + r * math.cos(2 * math.pi * k / N),
             iy + r * math.sin(2 * math.pi * k / N)) for k in range(N)]


def main():
    N, A, B, C = read_tri()
    rng = random.Random(1234567)
    R = 900
    # start from the full-inradius regular N-gon (a solid fallback)
    best_pts = full_incircle(A, B, C, N)
    best_val = min_area(best_pts)
    for _ in range(R):
        pts = [sample(A, B, C, rng) for _ in range(N)]
        v = min_area(pts)
        if v > best_val:
            best_val = v
            best_pts = pts
    out = ["%.12f %.12f" % (p[0], p[1]) for p in best_pts]
    sys.stdout.write("\n".join(out) + "\n")


main()
