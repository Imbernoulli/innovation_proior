# TIER: strong
# Farthest-point seed on a dense grid, then seeded local-search hill climbing:
# repeatedly relocate one endpoint of the closest pair to a feasible spot that
# increases the minimum pairwise distance. Multi-restart, deterministic.
# Uses incremental move evaluation: moving one point only needs O(n) work given
# the precomputed min-distance among the other points.
import sys, math, random


def read_instance():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it)); c = int(next(it))
    disks = [(float(next(it)), float(next(it)), float(next(it))) for _ in range(c)]
    return n, disks


def feasible(x, y, disks):
    if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
        return False
    for (cx, cy, r) in disks:
        if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
            return False
    return True


def min_pair_dist2(pts):
    m = float("inf"); bi = bj = 0
    n = len(pts)
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            dd = (xi - pts[j][0]) ** 2 + (yi - pts[j][1]) ** 2
            if dd < m:
                m = dd; bi = i; bj = j
    return m, bi, bj


def rest_min_excluding(pts, k):
    """min pairwise squared distance over pairs not involving index k."""
    m = float("inf"); n = len(pts)
    for i in range(n):
        if i == k:
            continue
        xi, yi = pts[i]
        for j in range(i + 1, n):
            if j == k:
                continue
            dd = (xi - pts[j][0]) ** 2 + (yi - pts[j][1]) ** 2
            if dd < m:
                m = dd
    return m


def dist2_to_others(pts, k, x, y):
    m = float("inf"); n = len(pts)
    for q in range(n):
        if q == k:
            continue
        dd = (x - pts[q][0]) ** 2 + (y - pts[q][1]) ** 2
        if dd < m:
            m = dd
    return m


def farthest_point_seed(n, disks, G):
    cand = []
    for i in range(G):
        for j in range(G):
            x = i / (G - 1); y = j / (G - 1)
            if feasible(x, y, disks):
                cand.append((x, y))
    chosen = [min(cand, key=lambda p: (p[0] + p[1]))]
    dist2 = [(px - chosen[0][0]) ** 2 + (py - chosen[0][1]) ** 2 for (px, py) in cand]
    while len(chosen) < n:
        bi = max(range(len(cand)), key=lambda kk: dist2[kk])
        chosen.append(cand[bi])
        cx, cy = cand[bi]
        for kk in range(len(cand)):
            dd = (cand[kk][0] - cx) ** 2 + (cand[kk][1] - cy) ** 2
            if dd < dist2[kk]:
                dist2[kk] = dd
    return chosen[:n]


def main():
    n, disks = read_instance()
    best = None
    best_m = -1.0

    for seedidx, G in enumerate((30, 45, 60)):
        pts = [list(p) for p in farthest_point_seed(n, disks, G)]
        rng = random.Random(1234 + 101 * seedidx)
        cur_m, bi, bj = min_pair_dist2(pts)
        for _ in range(3500):
            cur_m, bi, bj = min_pair_dist2(pts)
            k = bi if rng.random() < 0.5 else bj
            ox, oy = pts[k]
            rest = rest_min_excluding(pts, k)
            improved = False
            for _t in range(14):
                if _t < 7:
                    nx = rng.random(); ny = rng.random()
                else:
                    dk = dist2_to_others  # noqa
                    nn = None; nnd = float("inf")
                    for q in range(n):
                        if q == k:
                            continue
                        dd = (ox - pts[q][0]) ** 2 + (oy - pts[q][1]) ** 2
                        if dd < nnd:
                            nnd = dd; nn = q
                    dx = ox - pts[nn][0]; dy = oy - pts[nn][1]
                    nrm = math.hypot(dx, dy) or 1.0
                    step = 0.03 + 0.15 * rng.random()
                    nx = ox + dx / nrm * step + (rng.random() - 0.5) * 0.05
                    ny = oy + dy / nrm * step + (rng.random() - 0.5) * 0.05
                if not feasible(nx, ny, disks):
                    continue
                d_new = dist2_to_others(pts, k, nx, ny)
                new_m = rest if rest < d_new else d_new
                if new_m > cur_m + 1e-12:
                    pts[k] = [nx, ny]
                    cur_m = new_m
                    improved = True
                    break
            if not improved:
                q = rng.randrange(n)
                oxq, oyq = pts[q]
                restq = rest_min_excluding(pts, q)
                for _t in range(20):
                    nx = rng.random(); ny = rng.random()
                    if not feasible(nx, ny, disks):
                        continue
                    d_new = dist2_to_others(pts, q, nx, ny)
                    new_m = restq if restq < d_new else d_new
                    if new_m >= cur_m - 1e-12:
                        pts[q] = [nx, ny]
                        cur_m = new_m
                    break
        m, _, _ = min_pair_dist2(pts)
        if m > best_m:
            best_m = m
            best = [tuple(p) for p in pts]

    out = ["%.10f %.10f" % (x, y) for (x, y) in best]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
