import sys
from fractions import Fraction


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    inp = open(sys.argv[1]).read().split()
    out_raw = open(sys.argv[2]).read()

    try:
        it = iter(inp)
        G = int(next(it)); Tmin = int(next(it)); Tmax = int(next(it)); K = int(next(it))
        if G <= 0 or K <= 0 or Tmin <= 0 or Tmax < Tmin:
            fail("bad header")
        targets = []
        for _ in range(K):
            P = int(next(it)); Q = int(next(it)); lam = float(next(it))
            if P <= 0 or Q <= 0 or lam < 0:
                fail("bad target record")
            targets.append((P, Q, lam))
    except Exception:
        fail("bad input")

    out_toks = out_raw.split()
    expected = K * G * 2
    if len(out_toks) != expected:
        fail("token count mismatch: need %d got %d" % (expected, len(out_toks)))

    teeth = []
    for tok in out_toks:
        # strict base-10 integer literal only -- rejects "nan"/"inf"/floats/garbage up front,
        # never even calls int()/float() on a non-finite token.
        core = tok[1:] if tok[:1] == "-" else tok
        if core == "" or not core.isdigit():
            fail("non-integer token %r" % tok)
        teeth.append(int(tok))

    mid = (Tmin + Tmax) // 2
    norm = float(G * (Tmin + Tmax))

    idx = 0
    F_total = 0.0
    B_total = 0.0
    for (P, Q, lam) in targets:
        target = Fraction(P, Q)

        V = Fraction(1, 1)
        cost = 0
        for _s in range(G):
            a = teeth[idx]; b = teeth[idx + 1]; idx += 2
            if a < Tmin or a > Tmax or b < Tmin or b > Tmax:
                fail("teeth out of range: a=%d b=%d (range [%d,%d])" % (a, b, Tmin, Tmax))
            V *= Fraction(a, b)
            cost += a + b
        relerr = abs(V - target) / target
        F_total += float(relerr) + lam * (cost / norm)

        Vb = Fraction(1, 1)  # baseline: (mid, mid) every stage
        costb = G * 2 * mid
        relerrb = abs(Vb - target) / target
        B_total += float(relerrb) + lam * (costb / norm)

    B_total = max(1e-9, B_total)
    F_total = max(1e-9, F_total)
    sc = min(1000.0, 100.0 * B_total / F_total)
    print("F=%.9f B=%.9f Ratio: %.6f" % (F_total, B_total, sc / 1000.0))


if __name__ == "__main__":
    main()
