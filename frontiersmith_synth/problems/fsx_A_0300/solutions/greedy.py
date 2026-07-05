# TIER: greedy
# Enclose ALL the interior formation clusters at once: convex hull of every interior
# station (drops the 4 far corner stations 0..3).  Captures the clusters but pays a
# single big hull perimeter and cannot skip an expensive/low-value cluster -> a solid
# middle solution that beats the single-triangle reference but leaves value on the table.
import sys, json, math


def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def convex_hull_idx(pts, idxs):
    """Monotone-chain hull; returns hull as a list of ORIGINAL indices, ccw order,
    with collinear points dropped."""
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


def main():
    inst = json.load(sys.stdin)
    st = inst["stations"]
    n = len(st)
    interior = [i for i in range(n) if i >= 4]
    if len(interior) < 3:
        interior = list(range(n))
    hull = convex_hull_idx(st, interior)
    if len(hull) < 3:
        hull = [0, 1, 2]
    print(json.dumps({"tour": hull}))


main()
