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

    zones = []            # list of (w, t, lits)
    B = 0                 # baseline: weight cleared by all-production (all zero)
    Wtot = 0
    try:
        for _ in range(m):
            w = int(next(it)); k = int(next(it)); t = int(next(it))
            lits = [int(next(it)) for _ in range(k)]
            zones.append((w, t, lits))
            Wtot += w
            # under all-production (x=0), literal -v is satisfied, +v is not.
            neg = sum(1 for l in lits if l < 0)
            if neg >= t:
                B += w
    except Exception:
        fail("bad zone data")
    B = max(1, B)

    # ---- parse participant output: exactly n tokens, each 0 or 1 ----
    if len(out) != n:
        fail("expected %d values, got %d" % (n, len(out)))
    x = [0] * (n + 1)     # 1-indexed
    for i, tkn in enumerate(out):
        if tkn != "0" and tkn != "1":
            fail("value not 0/1: %r" % tkn)
        x[i + 1] = 1 if tkn == "1" else 0

    # ---- objective F: total weight of cleared zones (>= t satisfied literals) ----
    F = 0
    for w, t, lits in zones:
        c = 0
        for l in lits:
            if l > 0:
                if x[l] == 1:
                    c += 1
            else:
                if x[-l] == 0:
                    c += 1
        if c >= t:
            F += w

    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    print("F=%d B=%d W=%d Ratio: %.6f" % (F, B, Wtot, sc / 1000.0))

if __name__ == "__main__":
    main()
