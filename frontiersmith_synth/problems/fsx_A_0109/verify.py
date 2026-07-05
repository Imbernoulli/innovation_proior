#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic scorer (ans ignored).

Reads the belt instance (dimension, M, K anchor rigs), reads the
participant's N_free = M-K probe coordinates, forms the union of anchors +
probes, and computes the EXACT 2-D star discrepancy of that M-point set.

Objective = minimize star discrepancy F.
Internal baseline B = discrepancy of a trivial corner-grid probe layout
(union with the SAME anchors).  Minimization scoring:
    sc = min(1000, 100 * B / F);  print Ratio: sc/1000
-> trivial layout ~0.1, a 10x-better layout caps at 1.0.
"""
import sys, bisect, math

TOL = 1e-9


def die(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def star_discrepancy(pts):
    """Exact star discrepancy of pts in [0,1]^2 (O(n^2))."""
    n = len(pts)
    xs = sorted(set(p[0] for p in pts))
    ys = sorted(set(p[1] for p in pts))
    xr = {v: i for i, v in enumerate(xs)}
    yr = {v: i for i, v in enumerate(ys)}
    nx, ny = len(xs), len(ys)
    H = [[0] * ny for _ in range(nx)]
    for (px, py) in pts:
        H[xr[px]][yr[py]] += 1
    # P[a+1][b+1] = #{ px<=xs[a] and py<=ys[b] }
    P = [[0] * (ny + 1) for _ in range(nx + 1)]
    for a in range(nx):
        row = P[a + 1]; prow = P[a]; Ha = H[a]
        run = 0
        for b in range(ny):
            run += Ha[b]
            row[b + 1] = prow[b + 1] + run
    Xc = xs + ([1.0] if xs[-1] != 1.0 else [])
    Yc = ys + ([1.0] if ys[-1] != 1.0 else [])
    n_inv = 1.0 / n
    best = 0.0
    for vx in Xc:
        a_le = bisect.bisect_right(xs, vx) - 1
        a_lt = bisect.bisect_left(xs, vx) - 1
        rle = P[a_le + 1]; rlt = P[a_lt + 1]
        for vy in Yc:
            b_le = bisect.bisect_right(ys, vy) - 1
            b_lt = bisect.bisect_left(ys, vy) - 1
            vol = vx * vy
            pos = rle[b_le + 1] * n_inv - vol
            neg = vol - rlt[b_lt + 1] * n_inv
            if pos > best:
                best = pos
            if neg > best:
                best = neg
    return best


def corner_grid(nf):
    g = int(math.ceil(math.sqrt(nf)))
    return [((i % g) / g, (i // g) / g) for i in range(nf)]


def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    d = int(next(it)); M = int(next(it)); K = int(next(it))
    anchors = []
    for _ in range(K):
        x = float(next(it)); y = float(next(it))
        anchors.append((x, y))
    return d, M, K, anchors


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    d, M, K, anchors = read_instance(inf)
    nf = M - K

    # ---- parse participant probes ----
    try:
        toks = open(outf).read().split()
    except Exception:
        die("cannot read output")
    if len(toks) != 2 * nf:
        die("expected %d numbers (%d probes), got %d" % (2 * nf, nf, len(toks)))
    probes = []
    try:
        for i in range(nf):
            x = float(toks[2 * i]); y = float(toks[2 * i + 1])
            probes.append((x, y))
    except Exception:
        die("non-numeric coordinate")
    for (x, y) in probes:
        if not (math.isfinite(x) and math.isfinite(y)):
            die("non-finite coordinate")
        if x < -TOL or x > 1.0 + TOL or y < -TOL or y > 1.0 + TOL:
            die("coordinate outside [0,1]")

    # clamp tiny tolerance overruns into [0,1]
    probes = [(min(1.0, max(0.0, x)), min(1.0, max(0.0, y))) for (x, y) in probes]

    full = anchors + probes
    F = star_discrepancy(full)

    # internal baseline: trivial corner-grid probes + same anchors
    B = star_discrepancy(anchors + corner_grid(nf))

    if F <= 1e-12:
        sc = 1000.0
    else:
        sc = min(1000.0, 100.0 * B / F)
    print("F=%.6f B=%.6f  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
