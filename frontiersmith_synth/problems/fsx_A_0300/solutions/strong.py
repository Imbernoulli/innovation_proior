# TIER: strong
# Cluster-selection local search.  Group the formations into clusters, then SEARCH over
# which subset of clusters to enclose and how tightly to hug each (k nearest stations,
# several k).  Each candidate loop is scored with the true contest objective; the best
# valid loop wins.  This trades rope against documentation value -- skipping a far or
# low-value cluster when the rope to reach it costs more than it is worth -- and hugs
# the chosen clusters more tightly than the all-clusters hull, so it beats both the
# single triangle and the enclose-everything greedy.
import sys, json, math
from itertools import combinations


def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_seg(a, b, p):
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def seg_int(p1, p2, p3, p4):
    d1 = orient(p3, p4, p1); d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3); d4 = orient(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    if d1 == 0 and on_seg(p3, p4, p1): return True
    if d2 == 0 and on_seg(p3, p4, p2): return True
    if d3 == 0 and on_seg(p1, p2, p3): return True
    if d4 == 0 and on_seg(p1, p2, p4): return True
    return False


def is_simple(V):
    n = len(V)
    if n < 3:
        return False
    if len(set(map(tuple, V))) != n:
        return False
    for i in range(n):
        if orient(V[i], V[(i + 1) % n], V[(i + 2) % n]) == 0:
            return False
    for i in range(n):
        a1, a2 = V[i], V[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            b1, b2 = V[j], V[(j + 1) % n]
            if seg_int(a1, a2, b1, b2):
                return False
    return True


def strict_inside(p, V):
    n = len(V)
    for i in range(n):
        a, b = V[i], V[(i + 1) % n]
        if orient(a, b, p) == 0 and on_seg(a, b, p):
            return False
    x, y = p; inside = False
    for i in range(n):
        x1, y1 = V[i]; x2, y2 = V[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xint:
                inside = not inside
    return inside


def perim(V):
    s = 0.0; n = len(V)
    for i in range(n):
        ax, ay = V[i]; bx, by = V[(i + 1) % n]
        s += math.hypot(ax - bx, ay - by)
    return s


def convex_hull_idx(pts, idxs):
    items = sorted(set(idxs), key=lambda i: (pts[i][0], pts[i][1]))
    if len(items) < 3:
        return list(items)

    def build(seq):
        h = []
        for i in seq:
            while len(h) >= 2 and orient(pts[h[-2]], pts[h[-1]], pts[i]) <= 0:
                h.pop()
            h.append(i)
        return h

    lower = build(items)
    upper = build(items[::-1])
    return lower[:-1] + upper[:-1]


def value_of(tour, st, feats, lam):
    if len(tour) < 3 or len(set(tour)) != len(tour):
        return None
    V = [st[i] for i in tour]
    if not is_simple(V):
        return None
    val = 0
    for (fx, fy, fv) in feats:
        if strict_inside((fx, fy), V):
            val += fv
    return val - lam * perim(V)


def main():
    inst = json.load(sys.stdin)
    st = inst["stations"]; feats = inst["features"]; lam = inst["lam"]
    n = len(st)

    # ---- cluster the formations (single-linkage within a threshold) ----
    F = len(feats)
    parent = list(range(F))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    thr = 170
    for a in range(F):
        for b in range(a + 1, F):
            if abs(feats[a][0] - feats[b][0]) + abs(feats[a][1] - feats[b][1]) <= thr:
                parent[find(a)] = find(b)
    groups = {}
    for a in range(F):
        groups.setdefault(find(a), []).append(a)
    glist = list(groups.values())
    # keep the highest total-value groups (cap to keep subset search small)
    glist.sort(key=lambda g: -sum(feats[i][2] for i in g))
    glist = glist[:7]

    # centroid + nearest stations of each group
    gnear = []
    for g in glist:
        cx = sum(feats[i][0] for i in g) / len(g)
        cy = sum(feats[i][1] for i in g) / len(g)
        order = sorted(range(n), key=lambda s: (st[s][0] - cx) ** 2 + (st[s][1] - cy) ** 2)
        gnear.append(order)

    best_tour = [0, 1, 2]
    best_val = value_of(best_tour, st, feats, lam)
    if best_val is None:
        best_val = -1e18

    def consider(tour):
        nonlocal best_tour, best_val
        v = value_of(tour, st, feats, lam)
        if v is not None and v > best_val:
            best_val = v; best_tour = list(tour)

    # baseline candidates: best triangle and the enclose-everything interior hull
    interior = [i for i in range(n) if i >= 4] or list(range(n))
    consider(convex_hull_idx(st, interior))
    # a cheap best-triangle probe (top formations' neighbourhoods)
    for g in gnear[:3]:
        consider(convex_hull_idx(st, g[:3]))

    # ---- search: every subset of clusters x several hull tightness levels ----
    G = len(glist)
    ks = [6, 8, 10, 14]
    for r in range(1, G + 1):
        for subset in combinations(range(G), r):
            for k in ks:
                idxs = []
                for gi in subset:
                    idxs.extend(gnear[gi][:k])
                consider(convex_hull_idx(st, idxs))

    print(json.dumps({"tour": best_tour}))


main()
