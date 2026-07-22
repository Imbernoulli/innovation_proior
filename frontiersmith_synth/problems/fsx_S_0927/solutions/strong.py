# TIER: strong
# The insight: don't commit to one construction recipe -- DISPATCH on cheap
# structural probes computed straight from the public points (no family label
# is ever given), then apply a SHARED 2-opt polish that every branch reuses.
#
#   probe 1: coefficient of variation of each stop's nearest-neighbor distance
#            (cv = std/mean of the 1-NN distances). High cv -> the layout has
#            very uneven local density (radiating spokes: near-hub points are
#            packed, tips are isolated) -- a sign that plain nearest-neighbor
#            construction will get stranded chasing local density and pay a
#            huge, unavoidable jump back.
#   probe 2: convex-hull vertex fraction (|hull| / n). A small fraction means
#            most stops sit deep inside the hull in a few separated pockets
#            (multiple grid blocks / tight clusters) where a boustrophedon
#            (row-band) sweep keeps the loop from crossing between pockets
#            more than necessary.
#
# Routing rule (probes only, no family id):
#   cv > 1.15 and hull_frac < 0.35   -> angular sweep around the centroid
#   hull_frac < 0.22                 -> row-band (grid-strip) boustrophedon
#   otherwise                        -> nearest neighbor
# Every branch then gets the SAME bounded 2-opt local-search pass (shared
# preprocessing/polish all three branches reuse) before the tour is emitted.
import sys, json, math

inst = json.load(sys.stdin)
pts = [tuple(p) for p in inst["points"]]
n = inst["n"]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nn_tour():
    visited = [False] * n
    order = [0]
    visited[0] = True
    cur = 0
    for _ in range(n - 1):
        best, bd = -1, float("inf")
        for j in range(n):
            if not visited[j]:
                d = dist(pts[cur], pts[j])
                if d < bd:
                    bd, best = d, j
        order.append(best)
        visited[best] = True
        cur = best
    return order


def sweep_tour():
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    return sorted(range(n), key=lambda i: (math.atan2(pts[i][1] - cy, pts[i][0] - cx),
                                            dist(pts[i], (cx, cy))))


def gridstrip_tour(n_strips=6):
    ys = [p[1] for p in pts]
    ymin, ymax = min(ys), max(ys)
    h = (ymax - ymin) / n_strips if ymax > ymin else 1.0

    def key(i):
        strip = min(n_strips - 1, int((pts[i][1] - ymin) / h)) if h > 0 else 0
        x = pts[i][0]
        if strip % 2 == 1:
            x = -x
        return (strip, x)

    return sorted(range(n), key=key)


def two_opt(order, max_passes=6):
    order = list(order)
    m = len(order)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(m - 1):
            a, b = order[i], order[i + 1]
            for j in range(i + 2, m):
                if i == 0 and j == m - 1:
                    continue
                c, d = order[j], order[(j + 1) % m]
                old = dist(pts[a], pts[b]) + dist(pts[c], pts[d])
                new = dist(pts[a], pts[c]) + dist(pts[b], pts[d])
                if new < old - 1e-9:
                    order[i + 1:j + 1] = reversed(order[i + 1:j + 1])
                    improved = True
                    b = order[i + 1]
    return order


def variance_nn_dist():
    nnd = []
    for i in range(n):
        bd = float("inf")
        for j in range(n):
            if i != j:
                d = dist(pts[i], pts[j])
                if d < bd:
                    bd = d
        nnd.append(bd)
    mean = sum(nnd) / n
    var = sum((x - mean) ** 2 for x in nnd) / n
    cv = math.sqrt(var) / mean if mean > 0 else 0.0
    return cv


def hull_fraction():
    uniq = sorted(set(pts))
    if len(uniq) < 3:
        return 1.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in uniq:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(uniq):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return len(hull) / n


cv = variance_nn_dist()
hf = hull_fraction()

if cv > 1.15 and hf < 0.35:
    order = sweep_tour()
elif hf < 0.22:
    order = gridstrip_tour()
else:
    order = nn_tour()

order = two_opt(order, max_passes=6)
print(json.dumps({"tour": order}))
