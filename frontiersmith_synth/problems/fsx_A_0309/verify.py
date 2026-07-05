import sys
import math
import itertools

TOL = 1e-12

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def star_discrepancy(pts, d):
    """Exact L-infinity star discrepancy of a point set in [0,1]^d.

    The supremum of the local discrepancy over anchored boxes [0,a) is
    attained on the grid whose per-axis breakpoints are the point
    coordinates (plus the boundary 1.0). At each grid corner we evaluate
    both the closed box (points <= corner, drives the over-count) and the
    open box (points < corner, drives the under-count).
    """
    n = len(pts)
    axes = []
    for k in range(d):
        vals = sorted(set(p[k] for p in pts) | {1.0})
        axes.append(vals)
    best = 0.0
    for corner in itertools.product(*axes):
        vol = 1.0
        for c in corner:
            vol *= c
        closed = 0
        opencnt = 0
        for p in pts:
            le = True
            lt = True
            for k in range(d):
                if p[k] > corner[k] + TOL:
                    le = False
                    lt = False
                    break
                if not (p[k] < corner[k] - TOL):
                    lt = False
            if le:
                closed += 1
            if lt:
                opencnt += 1
        over = closed / n - vol
        under = vol - opencnt / n
        if over > best:
            best = over
        if under > best:
            best = under
    return best

def diagonal_baseline(M, d):
    return [tuple((i + 0.5) / M for _ in range(d)) for i in range(M)]

def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    # ---- parse instance ----
    try:
        d = int(inp[0])
        M = int(inp[1])
    except Exception:
        fail("bad input")
    if d < 1 or M < 1:
        fail("bad instance")

    # ---- internal baseline B: the "ranger diagonal" line of towers ----
    B = star_discrepancy(diagonal_baseline(M, d), d)
    B = max(B, 1e-9)

    # ---- parse participant output: exactly M*d finite floats in [0,1] ----
    need = M * d
    if len(out) != need:
        fail("expected %d numbers, got %d" % (need, len(out)))
    vals = []
    for tok in out:
        try:
            x = float(tok)
        except Exception:
            fail("non-numeric token %r" % tok)
        if not math.isfinite(x):
            fail("non-finite coordinate")
        if x < -TOL or x > 1.0 + TOL:
            fail("coordinate %r out of [0,1]" % x)
        # clamp tiny overshoot into range
        if x < 0.0:
            x = 0.0
        if x > 1.0:
            x = 1.0
        vals.append(x)

    pts = [tuple(vals[i * d:(i + 1) * d]) for i in range(M)]

    # ---- objective: star discrepancy of the submitted towers (minimize) ----
    F = star_discrepancy(pts, d)
    F = max(F, 1e-9)

    sc = min(1000.0, 100.0 * B / F)
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
