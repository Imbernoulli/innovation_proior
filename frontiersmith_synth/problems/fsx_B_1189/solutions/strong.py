# TIER: strong
"""The insight: a wall's image point must be geometrically consistent with
EVERY microphone simultaneously, not just the two used to triangulate it.
For every pair of given microphones and every pair of their (unlabeled,
possibly decoy-contaminated) readings, compute the (up to two) points that
would explain that pair -- then keep only candidates that are ALSO
consistent (within tolerance) with every OTHER given microphone's reading
list. A genuine wall's image point gets rediscovered this way from many
different microphone pairs and survives the full cross-check; a mislabeled
pairing (defeated by a per-mic rank swap) or a decoy reading essentially
never explains every remaining microphone at once, so it is filtered out.
This directly instantiates "geometric consistency across ALL microphone
pairs simultaneously prunes almost all assignments"."""
import sys
import math

TOL = 2e-3
CLUSTER_EPS = 5e-3


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def circle_intersections(Ma, Mb, ra, rb):
    ax, ay = Ma; bx, by = Mb
    dx, dy = bx - ax, by - ay
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return []
    if d > ra + rb + 1e-6 or d < abs(ra - rb) - 1e-6:
        return []
    a = (ra * ra - rb * rb + d * d) / (2 * d)
    h2 = ra * ra - a * a
    if h2 < 0:
        h2 = 0.0
    h = math.sqrt(h2)
    px, py = ax + a * dx / d, ay + a * dy / d
    perp_x, perp_y = -dy / d, dx / d
    if h < 1e-9:
        return [(px, py)]
    return [(px + h * perp_x, py + h * perp_y), (px - h * perp_x, py - h * perp_y)]


def support_count(P, mics, skip):
    cnt = 0
    for c, (pos, readings) in enumerate(mics):
        if c in skip:
            continue
        best = min(abs(dist(P, pos) - r) for r in readings)
        if best <= TOL:
            cnt += 1
    return cnt


def cluster(points_with_support, needed_min_others):
    """points_with_support: list of (P, support). Keep those with
    support>=needed_min_others, merge near-duplicates, return list of
    (centroid, best_support) sorted by support desc then multiplicity desc."""
    kept = [(P, s) for (P, s) in points_with_support if s >= needed_min_others]
    clusters = []  # list of [sum_x, sum_y, count, best_support]
    for P, s in kept:
        placed = False
        for cl in clusters:
            cx, cy = cl[0] / cl[2], cl[1] / cl[2]
            if dist((cx, cy), P) <= CLUSTER_EPS:
                cl[0] += P[0]; cl[1] += P[1]; cl[2] += 1; cl[3] = max(cl[3], s)
                placed = True
                break
        if not placed:
            clusters.append([P[0], P[1], 1, s])
    result = [((cl[0] / cl[2], cl[1] / cl[2]), cl[3], cl[2]) for cl in clusters]
    result.sort(key=lambda t: (-t[1], -t[2]))
    return result


def line_through_image(S, I):
    ix, iy = I[0] - S[0], I[1] - S[1]
    n = math.hypot(ix, iy)
    if n < 1e-6:
        ix, iy, n = 1.0, 0.0, 1.0
    mx, my = (S[0] + I[0]) / 2.0, (S[1] + I[1]) / 2.0
    perp_x, perp_y = -iy / n, ix / n
    return (mx + perp_x, my + perp_y, mx - perp_x, my - perp_y)


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    W = int(next(it)); K = int(next(it)); _tid = int(next(it))
    S = (float(next(it)), float(next(it)))
    mics = []
    for _ in range(K):
        mx = float(next(it)); my = float(next(it))
        L = int(next(it))
        readings = [float(next(it)) for _ in range(L)]
        mics.append(((mx, my), readings))

    candidates = []  # (point, support_over_remaining_mics)
    for i in range(K):
        for j in range(i + 1, K):
            Ma, ra_list = mics[i]
            Mb, rb_list = mics[j]
            skip = {i, j}
            for ra in ra_list:
                for rb in rb_list:
                    for P in circle_intersections(Ma, Mb, ra, rb):
                        s = support_count(P, mics, skip)
                        candidates.append((P, s))

    chosen = []
    max_possible = K - 2
    threshold = max_possible
    while threshold >= 0 and len(chosen) < W:
        clusters = cluster(candidates, threshold)
        for (P, s, mult) in clusters:
            if len(chosen) >= W:
                break
            if all(dist(P, q) >= 0.06 for q in chosen):
                chosen.append(P)
        threshold -= 1

    # fallback filler (should essentially never trigger): generic ring points
    k = 0
    while len(chosen) < W:
        theta = 2 * math.pi * k / max(1, W) + 0.37
        cand = (S[0] + (1.2 + 0.01 * k) * math.cos(theta),
                S[1] + (1.2 + 0.01 * k) * math.sin(theta))
        if all(dist(cand, q) >= 0.06 for q in chosen):
            chosen.append(cand)
        k += 1

    out = [str(W)]
    for I in chosen[:W]:
        x1, y1, x2, y2 = line_through_image(S, I)
        out.append("%.9f %.9f %.9f %.9f" % (x1, y1, x2, y2))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
