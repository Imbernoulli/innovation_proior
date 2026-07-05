import sys

TOL = 1e-6

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def star_discrepancy(points):
    # Exact 2-D star discrepancy over anchored boxes [0,q).
    # Grid of candidate corners = point coordinates (plus 1.0) per axis.
    # closed-count - vol  (maximize points, shrink volume)  uses q at point coords, <=
    # vol - open-count     (maximize volume, few points)    uses q at point coords/1, <
    n = len(points)
    if n == 0:
        return 1.0
    xs = sorted(set(p[0] for p in points) | {1.0})
    ys = sorted(set(p[1] for p in points) | {1.0})
    disc = 0.0
    for a in xs:
        for b in ys:
            vol = a * b
            cl = 0
            op = 0
            for (px, py) in points:
                if px <= a and py <= b:
                    cl += 1
                if px < a and py < b:
                    op += 1
            v1 = cl / n - vol
            v2 = vol - op / n
            if v1 > disc:
                disc = v1
            if v2 > disc:
                disc = v2
    return disc

def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    try:
        it = iter(inp)
        d = int(next(it))
        M = int(next(it))
        K = int(next(it))
        anchors = []
        for _ in range(K):
            ax = float(next(it)); ay = float(next(it))
            anchors.append((ax, ay))
    except Exception:
        fail("bad input")

    if d != 2:
        fail("unsupported dimension")

    # ---- parse participant output: exactly M points, 2 coords each ----
    try:
        vals = [float(v) for v in out]
    except Exception:
        fail("non-numeric output")
    if len(vals) != 2 * M:
        fail("expected %d coordinates, got %d" % (2 * M, len(vals)))
    pts = [(vals[2 * j], vals[2 * j + 1]) for j in range(M)]

    # ---- feasibility: all in [0,1]^2 ----
    for (px, py) in pts:
        if px < -TOL or px > 1 + TOL or py < -TOL or py > 1 + TOL:
            fail("point out of [0,1]^2")
    # clamp tiny float overshoot
    pts = [(min(1.0, max(0.0, px)), min(1.0, max(0.0, py))) for (px, py) in pts]

    # ---- feasibility: every anchor must be present (within TOL) ----
    for (ax, ay) in anchors:
        ok = False
        for (px, py) in pts:
            if abs(px - ax) <= 1e-4 and abs(py - ay) <= 1e-4:
                ok = True
                break
        if not ok:
            fail("missing required signature recipe (%.6f, %.6f)" % (ax, ay))

    # ---- internal baseline B: anchors + all free points clustered at the
    #      origin (bake the same corner recipe repeatedly) -- a trivially
    #      feasible but badly-covering plan with high star discrepancy. ----
    base = list(anchors)
    rem = M - K
    for _ in range(rem):
        base.append((0.0, 0.0))
    B = star_discrepancy(base)
    B = max(1e-9, B)

    # ---- objective (minimization): star discrepancy of submitted set ----
    F = star_discrepancy(pts)
    F = max(1e-9, F)

    sc = min(1000.0, 100.0 * B / F)
    print("disc=%.6f baseline=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
