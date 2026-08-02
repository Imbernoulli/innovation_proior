import sys, math


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")
    try:
        out_raw = open(sys.argv[2]).read()
    except Exception:
        fail("cannot read output")

    try:
        it = iter(inp)
        n = int(next(it))
        if n <= 0 or n > 5000:
            fail("bad n")
        m = [0] * n
        pref = [None] * n
        reqs = [None] * n  # reqs[i][v] = list of (target, lo, hi), 1-indexed v
        for i in range(n):
            mi = int(next(it))
            if mi <= 0 or mi > 200:
                fail("bad version count")
            m[i] = mi
            pref[i] = [0] * (mi + 1)
            reqs[i] = [None] * (mi + 1)
            for v in range(1, mi + 1):
                p = int(next(it))
                if p <= 0 or p > 100:
                    fail("bad preference")
                pref[i][v] = p
                r = int(next(it))
                if r < 0 or r > 50:
                    fail("bad requirement count")
                edges = []
                for _ in range(r):
                    j = int(next(it)); lo = int(next(it)); hi = int(next(it))
                    if j <= i or j >= n:
                        fail("requirement target must have a strictly larger index")
                    if lo < 1 or hi < lo:
                        fail("bad requirement range")
                    edges.append((j, lo, hi))
                reqs[i][v] = edges
        leftover = list(it)
        if leftover:
            fail("trailing garbage in input")
    except SystemExit:
        raise
    except Exception:
        fail("malformed input")

    # ---- parse participant output: exactly n tokens, each a valid version ----
    out_toks = out_raw.split()
    if len(out_toks) != n:
        fail("expected exactly n version tokens, got %d" % len(out_toks))
    chosen = [0] * n
    for i, tok in enumerate(out_toks):
        try:
            x = float(tok)
        except Exception:
            fail("version token not a number")
        if not math.isfinite(x):
            fail("non-finite version")
        r = round(x)
        if abs(x - r) > 1e-6:
            fail("version not integral")
        if r < 1 or r > m[i]:
            fail("version out of range for package %d" % i)
        chosen[i] = r

    # ---- feasibility: every requirement edge from the CHOSEN version must hold ----
    for i in range(n):
        v = chosen[i]
        for (j, lo, hi) in reqs[i][v]:
            cj = chosen[j]
            if cj < lo or cj > hi:
                fail("package %d@%d requires package %d in [%d,%d], got %d" %
                     (i, v, j, lo, hi, cj))

    # ---- objective ----
    F = sum(pref[i][chosen[i]] for i in range(n))

    # internal baseline B: the checker's own trivial reference -- install
    # every package at its OLDEST version (version 1). By construction every
    # requirement range attached to a version-1 source always contains 1, so
    # this is always globally feasible; it is a real, checkable, low-value
    # reference, not the optimum.
    B = sum(pref[i][1] for i in range(n))
    B = max(B, 1e-9)

    sc = min(1000.0, 100.0 * F / B)
    print("F=%.4f B=%.4f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
