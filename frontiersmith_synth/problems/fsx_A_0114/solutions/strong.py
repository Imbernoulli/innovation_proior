# TIER: strong
# Independent-point local search for the Heilbronn (max-min-triangle-area) placement.
# Seed from several spread configurations (rings of different radii + a barycentric
# lattice), then repeatedly relocate a vertex of the current minimum-area triangle to
# a feasible spot that increases the global minimum triangle area. Multi-restart,
# fully seeded / deterministic. Uses incremental evaluation: moving one vertex only
# needs the (precomputed) min over triples not involving it, plus O(n^2) new triples.
import sys, math, random


def cross(ox, oy, ax, ay, bx, by):
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def tri_area(p, q, r):
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


def read_instance():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it))
    A = (float(next(it)), float(next(it)))
    B = (float(next(it)), float(next(it)))
    C = (float(next(it)), float(next(it)))
    return n, A, B, C


def inside(P, A, B, C, orient_sign):
    d1 = orient_sign * cross(A[0], A[1], B[0], B[1], P[0], P[1])
    d2 = orient_sign * cross(B[0], B[1], C[0], C[1], P[0], P[1])
    d3 = orient_sign * cross(C[0], C[1], A[0], A[1], P[0], P[1])
    return d1 >= 0 and d2 >= 0 and d3 >= 0


def bary(A, B, C, u, v):
    return (A[0] + u * (B[0] - A[0]) + v * (C[0] - A[0]),
            A[1] + u * (B[1] - A[1]) + v * (C[1] - A[1]))


def min_all(pts):
    n = len(pts); m = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = tri_area(pts[i], pts[j], pts[k])
                if a < m:
                    m = a
    return m


def min_excluding(pts, x):
    n = len(pts); m = float("inf")
    for i in range(n):
        if i == x:
            continue
        for j in range(i + 1, n):
            if j == x:
                continue
            for k in range(j + 1, n):
                if k == x:
                    continue
                a = tri_area(pts[i], pts[j], pts[k])
                if a < m:
                    m = a
    return m


def min_involving(pts, x, P):
    n = len(pts); m = float("inf")
    for i in range(n):
        if i == x:
            continue
        for j in range(i + 1, n):
            if j == x:
                continue
            a = tri_area(P, pts[i], pts[j])
            if a < m:
                m = a
    return m


def min_triple_index(pts):
    n = len(pts); m = float("inf"); best = (0, 1, 2)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = tri_area(pts[i], pts[j], pts[k])
                if a < m:
                    m = a; best = (i, j, k)
    return m, best


def rand_inside(rng, A, B, C):
    u = rng.random(); v = rng.random()
    if u + v > 1.0:
        u, v = 1.0 - u, 1.0 - v
    return bary(A, B, C, u, v)


def seed_config(kind, n, A, B, C, rng):
    if kind < 3:
        # ring inside incircle
        a = math.hypot(B[0] - C[0], B[1] - C[1])
        b = math.hypot(C[0] - A[0], C[1] - A[1])
        c = math.hypot(A[0] - B[0], A[1] - B[1])
        per = a + b + c
        ix = (a * A[0] + b * B[0] + c * C[0]) / per
        iy = (a * A[1] + b * B[1] + c * C[1]) / per
        area = 0.5 * abs(cross(A[0], A[1], B[0], B[1], C[0], C[1]))
        r = 2.0 * area / per
        rad = (0.6 + 0.12 * kind) * r
        return [[ix + rad * math.cos(2 * math.pi * i / n),
                 iy + rad * math.sin(2 * math.pi * i / n)] for i in range(n)]
    # random spread
    return [list(rand_inside(rng, A, B, C)) for _ in range(n)]


def main():
    n, A, B, C = read_instance()
    orient = cross(A[0], A[1], B[0], B[1], C[0], C[1])
    osign = 1.0 if orient >= 0 else -1.0

    best_pts = None; best_m = -1.0

    for restart in range(4):
        rng = random.Random(9001 + 137 * restart)
        pts = [list(p) for p in seed_config(restart, n, A, B, C, rng)]
        cur = min_all(pts)
        stale = 0
        for _ in range(1400):
            m, (i, j, k) = min_triple_index(pts)
            cur = m
            moved = False
            # try relocating each of the three vertices of the min triangle
            order = [i, j, k]
            rng.shuffle(order)
            for x in order:
                rest = min_excluding(pts, x)
                if rest <= cur + 1e-15:
                    # even a perfect move is bounded by rest; only worth trying if rest>cur
                    if rest <= cur:
                        continue
                for _t in range(12):
                    P = rand_inside(rng, A, B, C)
                    mi = min_involving(pts, x, P)
                    newm = rest if rest < mi else mi
                    if newm > cur + 1e-14:
                        pts[x] = [P[0], P[1]]
                        cur = newm
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                stale += 1
                # random kick: relocate a random vertex to a random feasible spot if
                # it does not hurt (accept sideways) to escape local optima
                q = rng.randrange(n)
                rest = min_excluding(pts, q)
                P = rand_inside(rng, A, B, C)
                mi = min_involving(pts, q, P)
                newm = rest if rest < mi else mi
                if newm >= cur - 1e-15:
                    pts[q] = [P[0], P[1]]
                if stale > 60:
                    break
            else:
                stale = 0

        m = min_all(pts)
        if m > best_m:
            best_m = m
            best_pts = [tuple(p) for p in pts]

    # clamp any tiny numerical drift back inside (safety)
    out = ["%.10f %.10f" % (x, y) for (x, y) in best_pts]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
