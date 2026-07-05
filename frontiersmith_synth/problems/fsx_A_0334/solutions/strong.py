# TIER: strong
"""Structured spread + hill climb.

1. Farthest-point insertion on a fine grid: repeatedly add the grid candidate that
   maximizes the min distance to the current combined set (landmarks + chosen
   stations). This yields a compact, well-separated packing (large d_min).
2. Direct hill climb on the true objective U = d_min/d_max: perturb one station at
   a time with shrinking, seeded random moves and grid snaps, accepting any move
   that raises U.

Deterministic (seeded by instance size). Beats both the diagonal baseline and the
best-of-random greedy, with a distinct per-test layout."""
import sys
import math
import random


def dists(pts):
    m = len(pts)
    dmin = float("inf")
    dmax = 0.0
    for i in range(m):
        xi, yi = pts[i]
        for j in range(i + 1, m):
            dx = xi - pts[j][0]
            dy = yi - pts[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < dmin:
                dmin = d
            if d > dmax:
                dmax = d
    return dmin, dmax


def uratio(pts):
    dmin, dmax = dists(pts)
    if dmax <= 0:
        return 0.0
    return dmin / dmax


def main():
    tok = sys.stdin.read().split()
    idx = 0
    n = int(tok[idx]); idx += 1
    k = int(tok[idx]); idx += 1
    land = []
    for _ in range(k):
        land.append((float(tok[idx]), float(tok[idx + 1]))); idx += 2

    rng = random.Random(4242 + n)

    # ---- grid candidates ----
    G = 26
    grid = []
    for a in range(G):
        for b in range(G):
            grid.append((0.02 + 0.96 * a / (G - 1), 0.02 + 0.96 * b / (G - 1)))

    # ---- farthest-point insertion (max-min-distance packing) ----
    chosen = list(land)
    stations = []
    for _ in range(n):
        best_pt = None
        best_d = -1.0
        for gx, gy in grid:
            dmin = float("inf")
            for cx, cy in chosen:
                dd = (gx - cx) ** 2 + (gy - cy) ** 2
                if dd < dmin:
                    dmin = dd
            if dmin > best_d:
                best_d = dmin
                best_pt = (gx, gy)
        chosen.append(best_pt)
        stations.append(best_pt)

    def full():
        return land + stations

    cur = uratio(full())

    # ---- hill climb on U ----
    step = 0.10
    for it in range(1400):
        i = rng.randrange(n)
        ox, oy = stations[i]
        if rng.random() < 0.5:
            nx = min(1.0, max(0.0, ox + rng.uniform(-step, step)))
            ny = min(1.0, max(0.0, oy + rng.uniform(-step, step)))
        else:
            gx, gy = grid[rng.randrange(len(grid))]
            nx, ny = gx, gy
        stations[i] = (nx, ny)
        u = uratio(full())
        if u > cur + 1e-12:
            cur = u
        else:
            stations[i] = (ox, oy)
        if it % 200 == 199:
            step *= 0.8

    out = ["%.6f %.6f" % (x, y) for x, y in stations]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
