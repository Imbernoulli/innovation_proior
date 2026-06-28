#!/usr/bin/env python3
"""Deterministic local scorer for "Cable Layout" (rectilinear Steiner tree).

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single number: the score (an integer). A higher score is better.

Scoring rule (see context.md "Evaluation settings"):
  * The instance has n terminals with integer coordinates on the [0, SIDE] grid.
    A cable layout must electrically connect ALL n terminals using axis-aligned
    (rectilinear) wire segments; extra Steiner junction points are allowed.
  * A SOLUTION is a list of axis-aligned segments. The wires form a planar wire
    network; two segments are electrically joined wherever they share a point.
    The layout is FEASIBLE iff (a) every emitted segment is axis-aligned
    (horizontal or vertical) with integer endpoints inside the grid, and (b) all
    n terminals lie in a single connected component of the wire network.
  * Let W = the TOTAL ROUTED WIRE LENGTH, i.e. the length of the geometric union
    of all segments (collinear overlaps counted ONCE, so padding a wire with
    duplicate/overlapping segments cannot be gamed -- only the area actually
    covered by copper counts). Shorter W is better.
  * Let G = the wire length of the scorer's own deterministic reference layout:
    a rectilinear minimum spanning tree over the terminals under the L1
    (Manhattan) metric, each tree edge routed as an L-shape. The scorer
    recomputes G itself, so the baseline is reproducible and solver-independent.
  * FEASIBILITY FLOOR: if the output is not parseable, contains a non-axis-aligned
    or out-of-grid segment, or fails to connect all terminals into one component,
    the solution is INFEASIBLE and the score is 0.
  * SCORE = round(1_000_000 * G / W) for a feasible layout with W > 0.
    The rectilinear-MST reference scores exactly 1_000_000; a shorter layout
    (e.g. one that introduces Steiner points to share wire) scores strictly more;
    a longer layout scores less but stays positive. Infeasible -> 0.

The scorer is self-contained and deterministic: it does not trust the solver and
recomputes G itself. n <= 1 is a degenerate full-credit case (no wire needed).
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    side = int(next(it))
    xs = [0] * n
    ys = [0] * n
    for i in range(n):
        xs[i] = int(next(it))
        ys[i] = int(next(it))
    return n, side, xs, ys


def read_solution(path):
    """Return a list of (x1,y1,x2,y2) int tuples, or None on a parse error."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if not toks:
        return None
    it = iter(toks)
    try:
        m = int(next(it))
    except (StopIteration, ValueError):
        return None
    if m < 0:
        return None
    segs = []
    for _ in range(m):
        try:
            x1 = int(next(it)); y1 = int(next(it))
            x2 = int(next(it)); y2 = int(next(it))
        except (StopIteration, ValueError):
            return None
        segs.append((x1, y1, x2, y2))
    # Any trailing tokens beyond the declared m segments are an error.
    if next(it, None) is not None:
        return None
    return segs


def union_length(segments):
    """Total length of the geometric union of axis-aligned integer segments.

    Collinear overlaps are counted once. Horizontal and vertical wires are
    handled independently; a crossing of an H wire and a V wire shares a single
    point (measure zero) and is not double-subtracted.
    """
    # group horizontal segments by their y, vertical by their x
    h = {}  # y -> list of (xlo, xhi)
    v = {}  # x -> list of (ylo, yhi)
    for (x1, y1, x2, y2) in segments:
        if y1 == y2:  # horizontal (covers degenerate points too: handled below)
            a, b = (x1, x2) if x1 <= x2 else (x2, x1)
            if a != b:
                h.setdefault(y1, []).append((a, b))
        else:  # x1 == x2 guaranteed by feasibility check -> vertical
            a, b = (y1, y2) if y1 <= y2 else (y2, y1)
            if a != b:
                v.setdefault(x1, []).append((a, b))

    total = 0
    for line in (h, v):
        for _, ivs in line.items():
            ivs.sort()
            cur_lo, cur_hi = ivs[0]
            for lo, hi in ivs[1:]:
                if lo > cur_hi:
                    total += cur_hi - cur_lo
                    cur_lo, cur_hi = lo, hi
                else:
                    if hi > cur_hi:
                        cur_hi = hi
            total += cur_hi - cur_lo
    return total


class DSU:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def connects_all(n, xs, ys, segments):
    """True iff all n terminals are in one connected component of the wires.

    The wire network is treated as a geometric graph over SEGMENTS: two
    segments are electrically joined wherever they share at least one point.
    We union segments that touch and attach every terminal to the segments it
    lies on, then check all terminals share one component. Touch tests are
    exact on the integer grid:
      * H-V pair: touch iff the vertical's x lies in the horizontal's x-range
        AND the horizontal's y lies in the vertical's y-range (this captures
        endpoint touches, T-junctions, and proper interior crossings alike).
      * H-H pair (same y): touch iff their x-ranges overlap (collinear share).
      * V-V pair (same x): touch iff their y-ranges overlap.
    """
    if n <= 1:
        return True

    H = []  # (y, xlo, xhi, seg_id)
    V = []  # (x, ylo, yhi, seg_id)
    m = len(segments)
    for sid, (x1, y1, x2, y2) in enumerate(segments):
        if y1 == y2:
            a, b = (x1, x2) if x1 <= x2 else (x2, x1)
            H.append((y1, a, b, sid))
        else:
            a, b = (y1, y2) if y1 <= y2 else (y2, y1)
            V.append((x1, a, b, sid))

    dsu = DSU(m)

    # H-V crossings / touches
    for (hy, hxa, hxb, hid) in H:
        for (vx, vya, vyb, vid) in V:
            if hxa <= vx <= hxb and vya <= hy <= vyb:
                dsu.union(hid, vid)

    # H-H collinear overlaps: group by y, sort by xlo, sweep
    Hby = {}
    for rec in H:
        Hby.setdefault(rec[0], []).append(rec)
    for y, recs in Hby.items():
        recs.sort(key=lambda r: r[1])
        cur_hi = recs[0][2]
        cur_id = recs[0][3]
        for (yy, xa, xb, sid) in recs[1:]:
            if xa <= cur_hi:           # overlap or touch -> join
                dsu.union(cur_id, sid)
                if xb > cur_hi:
                    cur_hi = xb
                # keep cur_id as the representative of this merged run
            else:
                cur_hi = xb
                cur_id = sid

    # V-V collinear overlaps: group by x
    Vbx = {}
    for rec in V:
        Vbx.setdefault(rec[0], []).append(rec)
    for x, recs in Vbx.items():
        recs.sort(key=lambda r: r[1])
        cur_hi = recs[0][2]
        cur_id = recs[0][3]
        for (xx, ya, yb, sid) in recs[1:]:
            if ya <= cur_hi:
                dsu.union(cur_id, sid)
                if yb > cur_hi:
                    cur_hi = yb
            else:
                cur_hi = yb
                cur_id = sid

    # attach terminals: each terminal must lie on at least one segment, and we
    # union all segments it lies on (a terminal is a connection point).
    term_root = [None] * n
    for i in range(n):
        px, py = xs[i], ys[i]
        owner = -1
        for (hy, hxa, hxb, hid) in H:
            if py == hy and hxa <= px <= hxb:
                owner = hid
                break
        if owner < 0:
            for (vx, vya, vyb, vid) in V:
                if px == vx and vya <= py <= vyb:
                    owner = vid
                    break
        if owner < 0:
            return False  # terminal touches no wire -> not connected
        term_root[i] = owner

    root = dsu.find(term_root[0])
    for i in range(1, n):
        if dsu.find(term_root[i]) != root:
            return False
    return True


def rectilinear_mst_length(n, xs, ys):
    """L1 (Manhattan) minimum spanning tree total length over the terminals.

    Reproducible reference: Prim's algorithm on the complete graph with
    Manhattan edge weights. Each MST edge would be routed as an L-shape of
    length equal to its Manhattan distance, so the routed length of this
    reference layout equals the MST weight.
    """
    if n <= 1:
        return 0
    INF = float('inf')
    in_tree = [False] * n
    best = [INF] * n
    best[0] = 0
    total = 0
    for _ in range(n):
        u = -1
        bu = INF
        for j in range(n):
            if not in_tree[j] and best[j] < bu:
                bu = best[j]
                u = j
        in_tree[u] = True
        total += 0 if bu == INF else bu
        xu, yu = xs[u], ys[u]
        for j in range(n):
            if in_tree[j]:
                continue
            d = abs(xu - xs[j]) + abs(yu - ys[j])
            if d < best[j]:
                best[j] = d
    return total


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, side, xs, ys = read_instance(sys.argv[1])

    if n <= 1:
        # nothing to connect: any (even empty) layout is full credit
        print(1_000_000)
        return

    segs = read_solution(sys.argv[2])
    if segs is None:
        print(0)  # INFEASIBLE -> floored to 0
        return

    # feasibility: every segment axis-aligned, integer, inside the grid
    for (x1, y1, x2, y2) in segs:
        if not (x1 == x2 or y1 == y2):
            print(0)
            return
        if x1 == x2 and y1 == y2:
            # degenerate zero-length segment: harmless, contributes nothing,
            # but reject to keep outputs clean
            print(0)
            return
        for c in (x1, y1, x2, y2):
            if c < 0 or c > side:
                print(0)
                return

    if not connects_all(n, xs, ys, segs):
        print(0)  # not all terminals connected -> infeasible
        return

    W = union_length(segs)
    if W <= 0:
        # terminals not all coincident (n>=2 distinct) yet zero wire => infeasible
        print(0)
        return

    G = rectilinear_mst_length(n, xs, ys)
    score = int(round(1_000_000.0 * G / W))
    print(score)


if __name__ == "__main__":
    main()
