# TIER: greedy
# The obvious "one hub per cluster" heuristic: single-linkage cluster the organs by
# a fixed distance threshold, drop ONE hub at each cluster's demand-weighted
# centroid, and wire source -> hub -> organs (falling back to a direct source link
# if an obstacle blocks a tube).  This captures the per-cluster pooling, so it beats
# the star -- but it is a FLAT star of hubs: every cluster pays its own full haul
# back to the source, and the hub sits at the centroid, not the cost-optimal split
# point.  It never pools the long near-source corridor across clusters, leaving the
# biggest concave saving on the table.
import sys, math

A = 2.0 / 3.0
GEOM_EPS = 1e-7


def seg_hits_rect(ax, ay, bx, by, rect):
    x0, y0, x1, y1 = rect
    x0 += GEOM_EPS; y0 += GEOM_EPS; x1 -= GEOM_EPS; y1 -= GEOM_EPS
    if x1 <= x0 or y1 <= y0:
        return False
    dx = bx - ax; dy = by - ay
    p = [-dx, dx, -dy, dy]
    q = [ax - x0, x1 - ax, ay - y0, y1 - ay]
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-15:
            if qi < 0:
                return False
        else:
            r = qi / pi
            if pi < 0:
                if r > t1:
                    return False
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return False
                if r < t1:
                    t1 = r
    return t1 - t0 > 1e-12


def main():
    t = sys.stdin.read().split()
    it = iter(t)
    K = int(next(it)); M = int(next(it))
    Wm = float(next(it)); Wd = float(next(it))
    sx = float(next(it)); sy = float(next(it))
    org = [(sx, sy, 0.0)]
    for _ in range(K):
        x = float(next(it)); y = float(next(it)); d = float(next(it))
        org.append((x, y, d))
    rects = []
    for _ in range(M):
        x0 = float(next(it)); y0 = float(next(it))
        x1 = float(next(it)); y1 = float(next(it))
        rects.append((x0, y0, x1, y1))

    def clear(ax, ay, bx, by):
        for r in rects:
            if seg_hits_rect(ax, ay, bx, by, r):
                return False
        return True

    # single-linkage clustering by a fixed distance threshold
    THRESH = 45.0
    comp = list(range(K + 1))

    def find(a):
        while comp[a] != a:
            comp[a] = comp[comp[a]]; a = comp[a]
        return a

    for i in range(1, K + 1):
        for j in range(i + 1, K + 1):
            if math.hypot(org[i][0] - org[j][0], org[i][1] - org[j][1]) <= THRESH:
                comp[find(i)] = find(j)

    clusters = {}
    for i in range(1, K + 1):
        clusters.setdefault(find(i), []).append(i)

    parent = [0] * (K + 1)
    steiner = []
    for members in clusters.values():
        tw = sum(org[o][2] for o in members) or 1.0
        cx = sum(org[o][0] * org[o][2] for o in members) / tw
        cy = sum(org[o][1] * org[o][2] for o in members) / tw
        # only use the hub if the source->hub tube is clear; else star those organs
        if not clear(sx, sy, cx, cy):
            for o in members:
                parent[o] = 0
            continue
        jid = K + 1 + len(steiner)
        steiner.append((cx, cy))
        for o in members:
            if clear(cx, cy, org[o][0], org[o][1]):
                parent[o] = jid
            else:
                parent[o] = 0
    # drop hubs that ended up with no organ child
    used = set(parent[o] for o in range(1, K + 1) if parent[o] > K)
    remap = {0: 0}
    for i in range(1, K + 1):
        remap[i] = i
    new_st = []
    for old in range(K + 1, K + 1 + len(steiner)):
        if old in used:
            remap[old] = K + 1 + len(new_st)
            new_st.append(steiner[old - (K + 1)])
    out = [str(len(new_st))]
    for (jx, jy) in new_st:
        out.append("%.6f %.6f" % (jx, jy))
    for o in range(1, K + 1):                 # organ parents (nodes 1..K)
        out.append(str(remap[parent[o]]))
    for _ in new_st:                          # junction parents (nodes K+1..K+P) -> source
        out.append("0")
    sys.stdout.write("\n".join(out) + "\n")


main()
