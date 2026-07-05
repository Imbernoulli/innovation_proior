# TIER: strong
"""Targeted multi-start local search for the Heilbronn-type spread.

Each start: a random feasible layout, then repeatedly identify the CURRENT faintest
triangle and try to enlarge it by perturbing one of its three vertices (a random
displacement, kept only if it strictly increases the global minimum triangle area and
stays inside the field). Perturbation scale anneals down. Best layout over all starts
is emitted. Genuinely different per-instance behavior from best-of-random."""
import sys
import random


def area(a, b, c):
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def min_triangle(pts):
    """Return (min_area, (i,j,k)) achieving it."""
    n = len(pts)
    best = float("inf")
    tri = (0, 1, 2)
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            dxj = pts[j][0] - xi
            dyj = pts[j][1] - yi
            for k in range(j + 1, n):
                ar = 0.5 * abs(dxj * (pts[k][1] - yi) - (pts[k][0] - xi) * dyj)
                if ar < best:
                    best = ar
                    tri = (i, j, k)
    return best, tri


def inside(p):
    return p[0] >= 0.0 and p[1] >= 0.0 and (p[0] + p[1]) <= 1.0


def rand_layout(rng, n):
    pts = []
    for _ in range(n):
        a = rng.random()
        b = rng.random()
        if a + b > 1.0:
            a, b = 1.0 - a, 1.0 - b
        pts.append((a, b))
    return pts


def local_search(rng, pts, iters):
    cur, tri = min_triangle(pts)
    step = 0.30
    for t in range(iters):
        step = 0.30 * (1.0 - t / float(iters)) + 0.01
        v = tri[rng.randrange(3)]
        ox, oy = pts[v]
        nx = ox + rng.uniform(-step, step)
        ny = oy + rng.uniform(-step, step)
        if not inside((nx, ny)):
            continue
        pts[v] = (nx, ny)
        newv, newtri = min_triangle(pts)
        if newv > cur + 1e-15:
            cur, tri = newv, newtri
        else:
            pts[v] = (ox, oy)
    return cur


def main():
    n = int(sys.stdin.read().split()[0])
    rng = random.Random(97531)
    starts = 5
    iters = 260
    best_pts = None
    best_val = -1.0
    for _ in range(starts):
        pts = rand_layout(rng, n)
        v = local_search(rng, pts, iters)
        if v > best_val:
            best_val = v
            best_pts = [p for p in pts]
    out = ["%.17g %.17g" % (x, y) for (x, y) in best_pts]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
