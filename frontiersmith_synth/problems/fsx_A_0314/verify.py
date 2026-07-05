import sys, math

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

# center/radius of the internal ring baseline. MUST match solutions/trivial.py.
CX, CY, RB = 0.5, 0.5, 0.26

def tri_area(a, b, c):
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

def min_triangle(pts):
    m = len(pts)
    best = None
    for i in range(m):
        ai = pts[i]
        for j in range(i + 1, m):
            aj = pts[j]
            for k in range(j + 1, m):
                a = tri_area(ai, aj, pts[k])
                if best is None or a < best:
                    best = a
    return 0.0 if best is None else best

def ring_baseline(n):
    pts = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        pts.append((CX + RB * math.cos(t), CY + RB * math.sin(t)))
    return pts

def main():
    try:
        inp = open(sys.argv[1]).read().split()
        it = iter(inp)
        n = int(next(it))
        xmin = float(next(it)); xmax = float(next(it))
        ymin = float(next(it)); ymax = float(next(it))
    except Exception:
        fail("bad instance")

    # ---- parse participant output strictly ----
    try:
        toks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if len(toks) != 2 * n:
        fail("expected %d numbers, got %d" % (2 * n, len(toks)))
    vals = []
    for t in toks:
        try:
            v = float(t)
        except Exception:
            fail("non-numeric token %r" % t)
        if not math.isfinite(v):
            fail("non-finite value")
        vals.append(v)

    eps = 1e-9
    pts = []
    for i in range(n):
        x = vals[2 * i]; y = vals[2 * i + 1]
        if x < xmin - eps or x > xmax + eps or y < ymin - eps or y > ymax + eps:
            fail("turbine (%r,%r) outside field" % (x, y))
        pts.append((x, y))

    # ---- objective: thinnest wake triangle ----
    F = min_triangle(pts)

    # ---- internal baseline: equally spaced ring ----
    B = min_triangle(ring_baseline(n))
    B = max(B, 1e-12)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.10g B=%.10g Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
