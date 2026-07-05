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

    clauses = []          # list of (w, [(v,a), ...])
    B = 0                 # baseline: weight met by all-default (all zero)
    Wtot = 0
    try:
        for _ in range(m):
            w = int(next(it)); k = int(next(it))
            pairs = []
            has_default = False
            for _ in range(k):
                v = int(next(it)); a = int(next(it))
                pairs.append((v, a))
                if a == 0:
                    has_default = True
            clauses.append((w, pairs))
            Wtot += w
            if has_default:
                B += w
    except Exception:
        fail("bad requirement data")
    B = max(1, B)

    # ---- parse participant output: exactly n tokens, each integer in [0, D-1] ----
    if len(out) != n:
        fail("expected %d values, got %d" % (n, len(out)))
    x = [0] * (n + 1)     # 1-indexed
    for i, t in enumerate(out):
        try:
            val = int(t)
        except Exception:
            fail("value not an integer: %r" % t)
        if val < 0 or val > D - 1:
            fail("config %d out of range [0,%d]" % (val, D - 1))
        x[i + 1] = val

    # ---- objective F ----
    F = 0
    for w, pairs in clauses:
        met = False
        for (v, a) in pairs:
            if x[v] == a:
                met = True
                break
        if met:
            F += w

    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    print("F=%d B=%d W=%d Ratio: %.6f" % (F, B, Wtot, sc / 1000.0))

if __name__ == "__main__":
    main()
