# TIER: strong
"""The insight: a graph built purely from nearest-neighbour edges has no
long-range connectivity, so greedy search from a fixed entry set can get
stuck in whichever region it starts in. This solution deliberately spends
part of each node's degree budget on DIVERSIFIED edges -- points chosen to
be far from everything the node already reaches (farthest-point / max-min
diversification against its own near-neighbour set), even though those
edges are individually longer than the alternatives it skipped. The same
farthest-point idea picks the R entry points, by global max-min dispersion
over ALL points, so no single region can hog the entry set. Together these
give the search bridges to cross between otherwise-disconnected regions and
entries that already start near each of them."""
import sys


def d2(a, b):
    dx = a[0] - b[0]; dy = a[1] - b[1]
    return dx * dx + dy * dy


def farthest_point_sample(pts, k, start=0):
    n = len(pts)
    k = min(k, n)
    chosen = [start]
    dmin = [d2(pts[start], p) for p in pts]
    while len(chosen) < k:
        best_i, best_v = -1, -1
        for i in range(n):
            if dmin[i] > best_v:
                best_v, best_i = dmin[i], i
        chosen.append(best_i)
        for i in range(n):
            dd = d2(pts[best_i], pts[i])
            if dd < dmin[i]:
                dmin[i] = dd
    return chosen


def node_edges(i, pts, M):
    n = len(pts)
    if n - 1 <= M:
        return [j for j in range(n) if j != i]
    bridge_count = 1
    near_budget = M - bridge_count

    dists = sorted((d2(pts[i], pts[j]), j) for j in range(n) if j != i)
    near = [j for _, j in dists[:near_budget]]
    selected = set(near)

    candidates = [j for j in range(n) if j != i and j not in selected]
    dmin = {}
    for j in candidates:
        dm = d2(pts[i], pts[j])
        for s in near:
            dd = d2(pts[s], pts[j])
            if dd < dm:
                dm = dd
        dmin[j] = dm

    bridges = []
    for _ in range(bridge_count):
        if not candidates:
            break
        best_j, best_v = -1, -1
        for j in candidates:
            v = dmin[j]
            if v > best_v:
                best_v, best_j = v, j
        bridges.append(best_j)
        candidates.remove(best_j)
        for j in candidates:
            dd = d2(pts[best_j], pts[j])
            if dd < dmin[j]:
                dmin[j] = dd

    return near + bridges


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); M = int(next(it)); R = int(next(it))
    pts = []
    for _ in range(N):
        x = int(next(it)); y = int(next(it))
        pts.append((x, y))
    Q = int(next(it))
    for _ in range(Q):
        next(it); next(it)

    entries = farthest_point_sample(pts, R, start=0)

    out = []
    out.append(" ".join(str(e) for e in entries))
    for i in range(N):
        nbrs = node_edges(i, pts, M)
        out.append(str(len(nbrs)) + " " + " ".join(str(v) for v in nbrs))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
