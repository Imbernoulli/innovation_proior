import sys, math, itertools

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def star_disc(pts, d):
    """Exact star discrepancy of a point set in [0,1]^d.

    D* = sup over anchored boxes [0,c) of | count / n - vol(c) |.
    The supremum is attained on the finite grid whose k-th coordinate ranges over the
    k-th coordinates of the points (plus the value 1). For each grid corner c we evaluate
    both the 'volume overshoot' term (vol - #open/n) and the 'count overshoot' term
    (#closed/n - vol), where #open uses strict '<' and #closed uses '<='. The maximum
    over the whole grid equals D* exactly (Niederreiter / DEM enumeration).
    """
    n = len(pts)
    if n == 0:
        return 1.0
    cand = []
    for k in range(d):
        vals = sorted(set(p[k] for p in pts))
        vals.append(1.0)
        cand.append(vals)
    best = 0.0
    for corner in itertools.product(*cand):
        vol = 1.0
        for c in corner:
            vol *= c
        c_open = 0
        c_closed = 0
        for p in pts:
            le = True
            lt = True
            for k in range(d):
                pk = p[k]
                ck = corner[k]
                if pk > ck:
                    le = False
                    lt = False
                    break
                if not (pk < ck):
                    lt = False
            if le:
                c_closed += 1
            if lt:
                c_open += 1
        d1 = vol - c_open / n
        d2 = c_closed / n - vol
        if d1 > best:
            best = d1
        if d2 > best:
            best = d2
    return best

def main():
    inp = open(sys.argv[1]).read().split()
    try:
        d = int(inp[0]); M = int(inp[1])
    except Exception:
        fail("bad input")
    if d < 1 or M < 1:
        fail("bad instance")

    raw = open(sys.argv[2]).read().split()
    if len(raw) != d * M:
        fail("expected %d numbers, got %d" % (d * M, len(raw)))

    coords = []
    for tok in raw:
        try:
            v = float(tok)
        except Exception:
            fail("non-numeric token %r" % tok)
        if not math.isfinite(v):
            fail("non-finite coordinate")
        if v < 0.0 or v > 1.0:
            fail("coordinate out of [0,1]: %r" % v)
        coords.append(v)

    pts = [tuple(coords[i * d:(i + 1) * d]) for i in range(M)]

    F = star_disc(pts, d)

    # Internal baseline B: the diagonal ride-line ((i+0.5)/M, ..., (i+0.5)/M).
    # A valid but poorly-spread construction; its discrepancy calibrates trivial -> ~0.1.
    diag = [tuple(((i + 0.5) / M,) * d) for i in range(M)]
    B = star_disc(diag, d)
    B = max(B, 1e-9)

    # Minimization: smaller star discrepancy is better.
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
