#!/usr/bin/env python3
# Deterministic checker for "One Artery Feeds Every Organ" (format C, quality-metric).
#
# Vascular space-colonization network design.  The solver returns a flow tree,
# rooted at the arterial source, that spans every organ (sink).  Junction
# (Steiner) points may be added anywhere in the plane so the tree can branch in
# empty space.  Each edge carries the total demand of the organs downstream of
# it.  Its cost combines a Murray-law MATERIAL term (tube volume, concave in
# flow: radius ~ flow^(1/3) so cross-section ~ flow^(2/3)) and a linear DELIVERY
# term (demand-weighted transport length).  Obstacles are rectangles no tube may
# cross.  Objective: MINIMIZE total cost.
#
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1]; the harness greps the LAST Ratio.
import sys, math

ALPHA = 2.0 / 3.0   # Murray-law material exponent (cross-section ~ flow^(2/3))
GEOM_EPS = 1e-7     # strict-interior margin for obstacle crossing
MAX_STEINER_PER_SINK = 6   # bound on artifact size


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    K = int(next(it)); M = int(next(it))
    Wm = float(next(it)); Wd = float(next(it))
    sx = float(next(it)); sy = float(next(it))
    sinks = []
    for _ in range(K):
        x = float(next(it)); y = float(next(it)); d = float(next(it))
        sinks.append((x, y, d))
    rects = []
    for _ in range(M):
        x0 = float(next(it)); y0 = float(next(it))
        x1 = float(next(it)); y1 = float(next(it))
        rects.append((x0, y0, x1, y1))
    return K, M, Wm, Wd, (sx, sy), sinks, rects


def seg_hits_rect(ax, ay, bx, by, rect):
    """True if the open segment (a,b) passes through the strict interior of the
    axis-aligned rectangle. Liang-Barsky clip; boundary-touching is allowed."""
    x0, y0, x1, y1 = rect
    # shrink to strict interior so tangent/boundary contact does not count
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
                return False   # parallel & outside this slab
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
    return t1 - t0 > 1e-12   # a real sub-segment lies inside


def edge_length(px, py, qx, qy):
    return math.hypot(px - qx, py - qy)


def subtree_demands(parent, n_nodes, K, sinks):
    """flow_i = total organ demand in the subtree rooted at node i (node 0=source).
    Returns list flow[0..n_nodes-1]. Iterative accumulation up parent chains."""
    flow = [0.0] * n_nodes
    for i in range(1, K + 1):
        flow[i] = sinks[i - 1][2]
    # order nodes by depth (distance to root) descending, then push demand up.
    depth = [0] * n_nodes
    order = []
    for i in range(1, n_nodes):
        # walk to root computing depth (tree already validated acyclic)
        d = 0
        j = i
        while j != 0:
            j = parent[j]
            d += 1
        depth[i] = d
        order.append(i)
    order.sort(key=lambda i: depth[i], reverse=True)
    for i in order:
        flow[parent[i]] += flow[i]
    return flow


def main():
    try:
        K, M, Wm, Wd, src, sinks, rects = read_instance(sys.argv[1])
    except Exception:
        fail("bad instance")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    it = iter(otoks)
    try:
        P = int(next(it))
    except StopIteration:
        fail("empty output")
    except Exception:
        fail("bad steiner count")
    if P < 0 or P > MAX_STEINER_PER_SINK * K + 4:
        fail("steiner count out of range")

    # coordinates: node 0 = source, 1..K = organs, K+1..K+P = steiner points
    coords = [src] + [(s[0], s[1]) for s in sinks]
    for _ in range(P):
        try:
            x = float(next(it)); y = float(next(it))
        except Exception:
            fail("bad steiner coord")
        if not (math.isfinite(x) and math.isfinite(y)):
            fail("non-finite steiner coord")
        coords.append((x, y))

    n_nodes = K + P + 1
    parent = [0] * n_nodes  # parent[0] unused (root)
    for i in range(1, n_nodes):
        try:
            pp = int(next(it))
        except Exception:
            fail("missing parent for node %d" % i)
        if pp < 0 or pp >= n_nodes or pp == i:
            fail("parent out of range at node %d" % i)
        parent[i] = pp
    # reject trailing garbage that would signal a malformed artifact
    extra = list(it)
    if len(extra) > 0:
        fail("trailing tokens in output")

    # validate: every node reaches root 0 without a cycle
    for i in range(1, n_nodes):
        seen = 0
        j = i
        while j != 0:
            j = parent[j]
            seen += 1
            if seen > n_nodes:
                fail("cycle in parent structure at node %d" % i)

    # obstacle feasibility: no edge crosses any rectangle interior
    for i in range(1, n_nodes):
        ax, ay = coords[i]
        bx, by = coords[parent[i]]
        for r in rects:
            if seg_hits_rect(ax, ay, bx, by, r):
                fail("edge %d crosses an obstacle" % i)

    flow = subtree_demands(parent, n_nodes, K, sinks)

    # total cost
    F = 0.0
    for i in range(1, n_nodes):
        ax, ay = coords[i]
        bx, by = coords[parent[i]]
        L = edge_length(ax, ay, bx, by)
        f = flow[i]
        if f <= 0.0:
            continue  # a dead steiner branch carries no flow (still may exist)
        F += L * (Wm * (f ** ALPHA) + Wd * f)

    if not math.isfinite(F) or F <= 0.0:
        fail("degenerate network cost")

    # internal baseline B: the star (every organ wired straight to the source).
    B = 0.0
    for (x, y, d) in sinks:
        L = math.hypot(x - src[0], y - src[1])
        B += L * (Wm * (d ** ALPHA) + Wd * d)
    if B <= 0.0:
        B = 1e-9

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
