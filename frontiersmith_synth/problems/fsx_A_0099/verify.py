#!/usr/bin/env python3
# Deterministic checker for "Mountain Rescue Relays: 3D Relay Grid" (format C, minimize
# 3D star discrepancy). CLI: python3 verify.py <in> <out> <ans>  (ans ignored).
# Prints "... Ratio: <r>" with r in [0,1]. Any feasibility violation -> Ratio: 0.0.
import sys
import numpy as np

TOL = 1e-6


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def star_discrepancy_3d(pts, m):
    """Exact 3D star discrepancy over the induced corner grid.
    pts: (m,3) float array in [0,1]^3.  The supremum of
        max( Nc(q)/m - V(q) ,  V(q) - No(q)/m )
    over anchored boxes [0,q] is attained on the grid whose per-axis candidate
    coordinates are the point coordinates (plus 1). Vectorized with numpy."""
    P = np.asarray(pts, dtype=np.float64)
    grids = []
    idx_closed = []
    idx_open = []
    for a in range(3):
        col = P[:, a]
        g = np.unique(np.concatenate([col, np.array([1.0])]))
        grids.append(g)
        # closed count: point contributes to corner index i iff g[i] >= coord
        # first such i = searchsorted(g, coord, 'left')
        idx_closed.append(np.searchsorted(g, col, side="left"))
        # open count: contributes iff g[i] > coord  -> first i = searchsorted(g, coord, 'right')
        idx_open.append(np.searchsorted(g, col, side="right"))
    Lx, Ly, Lz = len(grids[0]), len(grids[1]), len(grids[2])

    # closed-count tensor: histogram of first-eligible index, then inclusive prefix-sum per axis
    Hc = np.zeros((Lx, Ly, Lz), dtype=np.int64)
    np.add.at(Hc, (idx_closed[0], idx_closed[1], idx_closed[2]), 1)
    Nc = Hc.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)

    Ho = np.zeros((Lx, Ly, Lz), dtype=np.int64)
    # open index can be == L (coord below every grid value handled naturally); clip to bounds
    io = [np.clip(idx_open[a], 0, [Lx, Ly, Lz][a] - 1) for a in range(3)]
    # A point strictly-less contributes only where g[i] > coord; if searchsorted('right')
    # exceeds the last index it never qualifies (no grid corner strictly greater within [0,1]).
    valid = (idx_open[0] < Lx) & (idx_open[1] < Ly) & (idx_open[2] < Lz)
    np.add.at(Ho, (io[0][valid], io[1][valid], io[2][valid]), 1)
    No = Ho.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)

    V = np.multiply.outer(np.multiply.outer(grids[0], grids[1]), grids[2])
    dplus = Nc / m - V
    dminus = V - No / m
    return float(max(dplus.max(), dminus.max()))


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
        m = int(itoks[0])
        d = int(itoks[1])
    except Exception:
        fail("bad instance")

    if d != 3:
        fail("unsupported dimension")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    need = 3 * m
    if len(otoks) != need:
        fail("expected exactly %d numbers (%d relays x 3), got %d" % (need, m, len(otoks)))

    pts = []
    for k in range(m):
        try:
            x = float(otoks[3 * k])
            y = float(otoks[3 * k + 1])
            z = float(otoks[3 * k + 2])
        except Exception:
            fail("bad relay %d" % k)
        for v in (x, y, z):
            if not np.isfinite(v):
                fail("non-finite relay %d" % k)
            if v < -TOL or v > 1.0 + TOL:
                fail("relay %d outside [0,1]^3" % k)
        pts.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y)), min(1.0, max(0.0, z))))

    F = star_discrepancy_3d(pts, m)

    # internal trivial baseline: m-relay main-diagonal set in the cube
    base = [((i + 0.5) / m,) * 3 for i in range(m)]
    B = star_discrepancy_3d(base, m)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
