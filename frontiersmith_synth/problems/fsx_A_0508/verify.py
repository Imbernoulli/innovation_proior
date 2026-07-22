import sys, math


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def settle(h, S):
    """Abelian angle-of-repose relaxation, walls at both ends (grains conserved).
    Move grains from a higher cell to its lower neighbour whenever the slope
    exceeds S; the stable state (every |h[i]-h[i+1]| <= S) is unique.
    Sum-of-squares strictly decreases per move -> guaranteed termination."""
    N = len(h)
    while True:
        stable = True
        for i in range(N - 1):
            d = h[i] - h[i + 1]
            if d > S:
                m = (d - S + 1) // 2
                h[i] -= m
                h[i + 1] += m
                stable = False
            elif -d > S:
                m = (-d - S + 1) // 2
                h[i + 1] -= m
                h[i] += m
                stable = False
        if stable:
            return


def main():
    # ---- read instance ----
    try:
        raw = open(sys.argv[1]).read().split()
        it = iter(raw)
        N = int(next(it)); K = int(next(it)); S = int(next(it))
        L = int(next(it)); G = int(next(it))
        t = [int(next(it)) for _ in range(N)]
    except Exception:
        fail("bad input")

    # ---- read participant artifact: exactly K pours "c g" ----
    otoks = open(sys.argv[2]).read().split()
    if len(otoks) != 2 * K:
        fail("expected %d tokens, got %d" % (2 * K, len(otoks)))
    pours = []
    for s in range(K):
        cs = otoks[2 * s]; gs = otoks[2 * s + 1]
        # reject non-finite / non-integer tokens explicitly
        for tok in (cs, gs):
            low = tok.lower()
            if ("nan" in low) or ("inf" in low) or ("." in tok) or ("e" in low):
                fail("non-integer token %r" % tok)
        try:
            c = int(cs); g = int(gs)
        except Exception:
            fail("parse pour %d" % s)
        if not math.isfinite(c) or not math.isfinite(g):
            fail("non-finite")
        if c < 0 or c >= N:
            fail("cell %d out of range" % c)
        if g < 0 or g > G:
            fail("grains %d out of [0,%d]" % (g, G))
        pours.append((c, g))

    # ---- replay physics; accumulate integrated overshoot ----
    h = [0] * N
    integ_over = 0
    for (c, g) in pours:
        h[c] += g
        settle(h, S)
        integ_over += sum(hi - ti for hi, ti in zip(h, t) if hi > ti)
    shortfall = sum(ti - hi for hi, ti in zip(h, t) if ti > hi)

    # Cost: final undershoot plus the overshoot AVERAGED over the K stages.
    # Both terms are scaled by K so the score is an exact integer ratio and the
    # stage count K cancels out of the normalised ratio (a longer schedule is not
    # intrinsically punished; only the FRACTION of time material sits above target
    # matters).  integ_over already sums overshoot over all K stages == K * mean.
    F = K * shortfall + L * integ_over      # cost to minimise (>=0 integer)

    # ---- internal baseline B = do-nothing feasible cost (pour zero grains) ----
    B = K * sum(t)                          # all shortfall every stage, no overshoot
    B = max(1, B)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d B=%d shortfall=%d integ_over=%d Ratio: %.6f"
          % (F, B, shortfall, integ_over, sc / 1000.0))


if __name__ == "__main__":
    main()
