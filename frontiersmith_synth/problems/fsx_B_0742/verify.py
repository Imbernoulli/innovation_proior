import sys, math

EPS = 1e-6


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def parse_instance(text):
    it = iter(text.split())
    try:
        N = int(next(it)); T = int(next(it))
        caps = []; ms = []; as_ = []; bs = []; fast = []
        for _ in range(N):
            caps.append(int(next(it)))
            ms.append(float(next(it)))
            as_.append(float(next(it)))
            bs.append(float(next(it)))
            fast.append(int(next(it)))
        J = int(next(it)) - 1
        D = [float(next(it)) for _ in range(T)]
    except Exception:
        fail("bad instance")
    return N, T, caps, ms, as_, bs, fast, J, D


def cost_unit(a, m, b, p):
    return a * (p - m) * (p - m) + b * p


def trivial_dispatch(D, N, caps, J):
    """Internal baseline B: park the swing unit J at zero (always trivially
    safe for the N-1 rule) and split demand among the rest proportional to
    capacity, ignoring efficiency curves entirely."""
    p = [0.0] * N
    rest = sum(caps[i] for i in range(N) if i != J)
    for i in range(N):
        p[i] = 0.0 if i == J else D * caps[i] / rest
    return p


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_text = open(sys.argv[1]).read()
    try:
        out_text = open(sys.argv[2]).read()
    except Exception:
        fail("no output")

    N, T, caps, ms, as_, bs, fast, J, D = parse_instance(in_text)

    # ---- parse participant artifact: expect exactly N*T finite numbers ----
    toks = out_text.split()
    if len(toks) != N * T:
        fail("token count mismatch: got %d want %d" % (len(toks), N * T))
    vals = []
    for tok in toks:
        try:
            v = float(tok)
        except Exception:
            fail("non-numeric token %r" % tok)
        if not math.isfinite(v):
            fail("non-finite value %r" % tok)
        vals.append(v)

    P = [vals[t * N:(t + 1) * N] for t in range(T)]

    # ---- feasibility ----
    for t in range(T):
        p = P[t]
        for i in range(N):
            if p[i] < -EPS or p[i] > caps[i] + EPS:
                fail("t=%d unit %d out of [0,cap]: %r" % (t, i, p[i]))
        if abs(sum(p) - D[t]) > 1e-4 * max(1.0, D[t]):
            fail("t=%d demand not met: got %r want %r" % (t, sum(p), D[t]))
        # N-1 contingency: if the swing unit J trips, the OTHER
        # reserve-eligible ("fast") units' spare capacity must cover J's
        # current output.
        reserve = sum((caps[i] - p[i]) for i in range(N) if fast[i] == 1 and i != J)
        if reserve < p[J] - 1e-4 * max(1.0, p[J]):
            fail("t=%d N-1 reserve violated: reserve=%r pJ=%r" % (t, reserve, p[J]))

    # ---- objective: total fuel across all time steps ----
    F = 0.0
    for t in range(T):
        for i in range(N):
            F += cost_unit(as_[i], ms[i], bs[i], P[t][i])

    # ---- internal baseline B ----
    B = 0.0
    for t in range(T):
        pb = trivial_dispatch(D[t], N, caps, J)
        for i in range(N):
            B += cost_unit(as_[i], ms[i], bs[i], pb[i])
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
