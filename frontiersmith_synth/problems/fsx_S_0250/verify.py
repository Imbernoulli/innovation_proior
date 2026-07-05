import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    it = iter(inp)
    try:
        n = int(next(it)); m = int(next(it))
    except Exception:
        fail("bad header")

    clauses = []          # list of (w, lits)
    B = 0                 # baseline weight recovered by the all-shallow plan (all zeros)
    Wtot = 0
    try:
        for _ in range(m):
            w = int(next(it)); k = int(next(it))
            if w < 1 or k < 1:
                fail("bad clause meta")
            lits = [int(next(it)) for _ in range(k)]
            for l in lits:
                if l == 0 or abs(l) > n:
                    fail("bad literal")
            clauses.append((w, lits))
            Wtot += w
            # all-shallow (x=0): a positive literal +v is false, a negative literal
            # -v is true. Clause recovered by baseline iff it has >=1 negative literal.
            if any(l < 0 for l in lits):
                B += w
    except Exception:
        fail("bad clause data")
    B = max(1, B)

    # ---- parse participant plan: exactly n tokens, each 0 or 1 ----
    if len(out) != n:
        fail("expected %d values, got %d" % (n, len(out)))
    x = [0] * (n + 1)     # 1-indexed
    for i, t in enumerate(out):
        if t != "0" and t != "1":
            fail("value not 0/1: %r" % (t,))
        x[i + 1] = 1 if t == "1" else 0

    # ---- objective F: total recovered artifact value ----
    F = 0
    for w, lits in clauses:
        sat = False
        for l in lits:
            if l > 0:
                if x[l] == 1:
                    sat = True; break
            else:
                if x[-l] == 0:
                    sat = True; break
        if sat:
            F += w

    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    print("F=%d B=%d W=%d Ratio: %.6f" % (F, B, Wtot, sc / 1000.0))

if __name__ == "__main__":
    main()
