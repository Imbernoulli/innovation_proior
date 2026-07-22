#!/usr/bin/env python3
"""Generator for fsx_A_0943 -- Star Atlas Portfolio.
Usage: gen.py <testId>   (1..10). Deterministic given testId. Prints one test to stdout.
"""
import sys, random

def emit(points, K):
    # points: list of (x,y,c) with c in [0,K-1]
    N = len(points)
    out = []
    out.append(f"{N} {K}")
    for (x, y, c) in points:
        out.append(f"{x} {y} {c}")
    sys.stdout.write("\n".join(out) + "\n")


def cell_centers(K, rnd, canvas, margin):
    """Place K non-overlapping square cells on a grid, shuffled, each of side `side`."""
    cols = 1
    while cols * cols < K:
        cols += 1
    rows = (K + cols - 1) // cols
    side = canvas // max(cols, rows)
    cells = []
    for r in range(rows):
        for c in range(cols):
            if len(cells) >= K:
                break
            cells.append((c * side, r * side, side))
    rnd.shuffle(cells)
    return cells[:K], side


def gen_cluster_points(rnd, cx0, cy0, side, margin, s):
    pts = []
    seen = set()
    tries = 0
    while len(pts) < s and tries < 20000:
        tries += 1
        x = rnd.randint(cx0 + margin, cx0 + side - margin)
        y = rnd.randint(cy0 + margin, cy0 + side - margin)
        if (x, y) in seen:
            continue
        seen.add((x, y))
        pts.append((x, y))
    while len(pts) < s:  # extremely unlikely fallback
        x = rnd.randint(cx0, cx0 + side)
        y = rnd.randint(cy0, cy0 + side)
        if (x, y) not in seen:
            seen.add((x, y)); pts.append((x, y))
    return pts


def gen_path_shape(rnd, cx0, cy0, side, margin, s):
    """Points roughly along a slightly wiggly line -- natural PATH affinity."""
    x0 = rnd.randint(cx0 + margin, cx0 + side - margin)
    y0 = rnd.randint(cy0 + margin, cy0 + side - margin)
    ang = rnd.uniform(0, 6.283)
    dx, dy = 0.6 * (side - 2 * margin) / max(1, s), 0
    pts = []
    seen = set()
    import math
    ux, uy = math.cos(ang), math.sin(ang)
    for i in range(s):
        jitter = rnd.randint(-side // 20 - 1, side // 20 + 1)
        px = int(x0 + ux * i * dx + (-uy) * jitter)
        py = int(y0 + uy * i * dx + (ux) * jitter)
        px = min(max(px, cx0 + 1), cx0 + side - 1)
        py = min(max(py, cy0 + 1), cy0 + side - 1)
        while (px, py) in seen:
            px += 1
        seen.add((px, py))
        pts.append((px, py))
    return pts


def gen_star_shape(rnd, cx0, cy0, side, margin, s):
    """Points in a tight ring around a center -- natural STAR affinity."""
    import math
    cx = cx0 + side // 2
    cy = cy0 + side // 2
    R = (side - 2 * margin) // 2
    pts = []
    seen = set()
    for i in range(s):
        a = 2 * math.pi * i / s + rnd.uniform(-0.2, 0.2)
        r = R * rnd.uniform(0.6, 1.0)
        px = int(cx + r * math.cos(a))
        py = int(cy + r * math.sin(a))
        while (px, py) in seen:
            px += 1
        seen.add((px, py))
        pts.append((px, py))
    return pts


def gen_zigzag_shape(rnd, cx0, cy0, side, margin, s):
    """Points along a sawtooth -- natural ZIGZAG affinity."""
    x0 = cx0 + margin
    width = side - 2 * margin
    amp = width // 3
    pts = []
    seen = set()
    for i in range(s):
        px = int(x0 + width * i / max(1, s - 1))
        py = cy0 + side // 2 + (amp if i % 2 == 0 else -amp) + rnd.randint(-side // 25 - 1, side // 25 + 1)
        py = min(max(py, cy0 + 1), cy0 + side - 1)
        while (px, py) in seen:
            px += 1
        seen.add((px, py))
        pts.append((px, py))
    return pts


def main():
    testId = int(sys.argv[1])
    rnd = random.Random(0x51A2 + 1009 * testId)

    if testId == 1:
        K, sizes, canvas, margin = 3, [6, 6, 6], 300, 15
        kind = "uniform"
    elif testId == 2:
        K, sizes, canvas, margin = 4, None, 500, 15
        kind = "uniform"
    elif testId == 3:
        K, sizes, canvas, margin = 5, None, 600, 18
        kind = "uniform"
    elif testId == 4:
        K, sizes, canvas, margin = 6, None, 800, 20
        kind = "uniform"
    elif testId == 5:
        K, sizes, canvas, margin = 6, None, 1000, 25
        kind = "planted"
    elif testId == 6:
        K, sizes, canvas, margin = 8, None, 1200, 28
        kind = "planted"
    elif testId == 7:
        K, sizes, canvas, margin = 10, None, 1500, 30
        kind = "trap"
    elif testId == 8:
        K, sizes, canvas, margin = 12, None, 1800, 30
        kind = "trap"
    elif testId == 9:
        K, sizes, canvas, margin = 10, None, 1800, 30
        kind = "needle"
    else:  # 10
        K, sizes, canvas, margin = 14, None, 2200, 32
        kind = "large_trap"

    cells, side = cell_centers(K, rnd, canvas, margin)

    if sizes is None:
        lo, hi = (8, 16) if testId <= 6 else (12, 24) if testId != 10 else (14, 20)
        sizes = [rnd.randint(lo, hi) for _ in range(K)]

    points = []
    for ci in range(K):
        cx0, cy0, _ = cells[ci]
        s = sizes[ci]
        if kind == "planted":
            shape = ci % 3
            if shape == 0:
                pts = gen_path_shape(rnd, cx0, cy0, side, margin, s)
            elif shape == 1:
                pts = gen_star_shape(rnd, cx0, cy0, side, margin, s)
            else:
                pts = gen_zigzag_shape(rnd, cx0, cy0, side, margin, s)
        elif kind == "needle" and ci == 0:
            pts = gen_star_shape(rnd, cx0, cy0, side, margin, s)  # a "perfect" star needle
        else:
            pts = gen_cluster_points(rnd, cx0, cy0, side, margin, s)
        for (x, y) in pts:
            points.append((x, y, ci))

    # trap/large_trap: genuinely INTERLEAVE two clusters point-by-point inside one shared
    # cell (a dense regular grid, jittered, then split into the two clusters by a random
    # shuffle) so a same-label nearest neighbor is usually NOT the true spatial nearest
    # neighbor. This is a secondary stress test for the crossing-freeness term (a
    # geometry-blind per-cluster construction occasionally has to route an edge past
    # points of the other cluster); it is deliberately NOT the main trap channel -- the
    # main trap (per the family's innovation hook) is that no per-cluster construction,
    # however good, can create PORTFOLIO-level shape diversity, which is what actually
    # separates strong from greedy on every test (see chk.py's `div` term).
    if kind in ("trap", "large_trap") and K >= 4:
        import math as _math
        c_a, c_b = 1, 2
        cx0, cy0, _ = cells[c_a]
        sA, sB = sizes[c_a], sizes[c_b]
        M = sA + sB
        cols = max(1, int(_math.ceil(_math.sqrt(M))))
        rows = max(1, -(-M // cols))
        cellw = max(2, (side - 2 * margin) // cols)
        cellh = max(2, (side - 2 * margin) // rows)
        grid = []
        for r in range(rows):
            for c in range(cols):
                if len(grid) >= M:
                    break
                jx = rnd.randint(-max(1, cellw // 6), max(1, cellw // 6))
                jy = rnd.randint(-max(1, cellh // 6), max(1, cellh // 6))
                x = cx0 + margin + c * cellw + cellw // 2 + jx
                y = cy0 + margin + r * cellh + cellh // 2 + jy
                x = min(max(x, cx0 + 1), cx0 + side - 1)
                y = min(max(y, cy0 + 1), cy0 + side - 1)
                grid.append((x, y))
        rnd.shuffle(grid)  # random label split -> same-cluster points are NOT mutual
                           # nearest neighbors, they're scattered among the other cluster
        ptsA, ptsB = grid[:sA], grid[sA:sA + sB]
        points = [(x, y, c) for (x, y, c) in points if c not in (c_a, c_b)]
        for (x, y) in ptsA:
            points.append((x, y, c_a))
        for (x, y) in ptsB:
            points.append((x, y, c_b))

    # de-duplicate any accidental exact coordinate collisions (must stay deterministic)
    seen_xy = set()
    final = []
    bump = 0
    for (x, y, c) in points:
        while (x, y) in seen_xy:
            x += 1
            bump += 1
        seen_xy.add((x, y))
        final.append((x, y, c))

    # stable re-order: keep original relative order (do not sort) so cluster id sequence
    # in the file is interleaved by construction order, not artificially grouped.
    rnd.shuffle(final) if False else None  # (kept ungrouped naturally: emitted per-cluster below)

    # Interleave points across clusters in a fixed deterministic pattern (round robin) so a
    # "connect points in file order" baseline does NOT trivially get a within-cluster chain
    # for free, and to avoid accidentally grouping by cluster in the file.
    by_cluster = {}
    for (x, y, c) in final:
        by_cluster.setdefault(c, []).append((x, y, c))
    order = []
    idxs = {c: 0 for c in by_cluster}
    active = list(by_cluster.keys())
    while active:
        nxt = []
        for c in active:
            i = idxs[c]
            if i < len(by_cluster[c]):
                order.append(by_cluster[c][i])
                idxs[c] += 1
                nxt.append(c)
        active = nxt

    emit(order, K)


if __name__ == "__main__":
    main()
