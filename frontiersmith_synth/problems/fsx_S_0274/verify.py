import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    it = iter(inp)
    try:
        n = int(next(it)); m = int(next(it)); D = int(next(it))
    except Exception:
        fail("bad header")

    clauses = []      # list of (w, [(v, a), ...])
    B = 0             # baseline: weight established by all-channel-1 tuning
    Wtot = 0
    try:
        for _ in range(m):
            w = int(next(it)); k = int(next(it))
            lits = []
            names_ch1 = False
            for _ in range(k):
                v = int(next(it)); a = int(next(it))
                lits.append((v, a))
                if a == 1:
                    names_ch1 = True
            clauses.append((w, lits))
            Wtot += w
            if names_ch1:
                B += w
    except Exception:
        fail("bad clause data")
    B = max(1, B)

    # ---- parse participant output: exactly n integer tokens, each in [1, D] ----
    if len(out) != n:
        fail("expected %d values, got %d" % (n, len(out)))
    x = [0] * (n + 1)   # 1-indexed
    for i, t in enumerate(out):
        try:
            val = int(t)
        except Exception:
            fail("non-integer channel: %r" % t)
        if val < 1 or val > D:
            fail("channel %d out of range [1,%d]" % (val, D))
        x[i + 1] = val

    # ---- objective F ----
    F = 0
    for w, lits in clauses:
        sat = False
        for (v, a) in lits:
            if x[v] == a:
                sat = True
                break
        if sat:
            F += w

    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    print("F=%d B=%d W=%d Ratio: %.6f" % (F, B, Wtot, sc / 1000.0))

if __name__ == "__main__":
    main()
