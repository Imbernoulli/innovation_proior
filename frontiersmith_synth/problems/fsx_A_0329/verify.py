#!/usr/bin/env python3
# Deterministic checker for the 3D Telescope Constellation problem
# (format C, minimize the L-infinity star discrepancy in [0,1]^3).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1].  Exact over the induced corner grid.
import sys
import numpy as np

TOL = 1e-6


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def star_discrepancy(pts, n):
    """Exact d-dimensional star discrepancy over the induced corner grid.
    pts: numpy array (n, d) with entries in [0,1].
    The supremum of both one-sided discrepancies is attained on the finite grid
    whose per-axis coordinates are the point coordinates in that axis, plus 1.0
    (closed corners use the point coordinate inclusively; open corners exclude it).
    """
    n_pts, d = pts.shape
    axes = []
    for j in range(d):
        vals = np.unique(np.concatenate([pts[:, j], np.array([1.0])]))
        axes.append(vals)
    grids = np.meshgrid(*axes, indexing="ij")
    corners = np.stack([g.ravel() for g in grids], axis=1)  # (C, d)
    C = corners.shape[0]
    V = np.prod(corners, axis=1)  # (C,)
    best = 0.0
    # chunk over corners to bound memory (c * n * d elements per block)
    chunk = max(1, 3_000_000 // max(1, n_pts * d))
    for s in range(0, C, chunk):
        cc = corners[s:s + chunk]                       # (c, d)
        ge = cc[:, None, :] >= pts[None, :, :]          # q >= p  (closed)
        gt = cc[:, None, :] > pts[None, :, :]           # q >  p  (open)
        Nc = np.all(ge, axis=2).sum(axis=1)             # closed count
        No = np.all(gt, axis=2).sum(axis=1)             # open count
        v = V[s:s + chunk]
        dplus = Nc / n - v
        dminus = v - No / n
        loc = float(np.maximum(dplus, dminus).max())
        if loc > best:
            best = loc
    return best


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
        n = int(itoks[0])
        d = int(itoks[1])
    except Exception:
        fail("bad instance")

    if d != 3:
        fail("unsupported dimension")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    need = 3 * n
    if len(otoks) != need:
        fail("expected exactly %d numbers (%d points), got %d" % (need, n, len(otoks)))

    pts = np.empty((n, 3), dtype=np.float64)
    for k in range(n):
        for j in range(3):
            try:
                val = float(otoks[3 * k + j])
            except Exception:
                fail("bad coordinate at point %d" % k)
            if not np.isfinite(val):
                fail("non-finite coordinate at point %d" % k)
            if val < -TOL or val > 1.0 + TOL:
                fail("point %d outside [0,1]^3" % k)
            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0
            pts[k, j] = val

    F = star_discrepancy(pts, n)

    # internal trivial baseline: n-point main-diagonal set in the cube
    base = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        t = (i + 0.5) / n
        base[i, 0] = t
        base[i, 1] = t
        base[i, 2] = t
    B = star_discrepancy(base, n)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
