import sys

SCALE = 1000


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def read_instance(path):
    try:
        toks = open(path).read().split()
    except Exception:
        fail("cannot read input")
    it = iter(toks)
    try:
        N = int(next(it)); K = int(next(it))
        ALPHA = int(next(it)); D = int(next(it))
        CC_NUM = int(next(it)); CC_DEN = int(next(it)); CENTER = int(next(it))
        if N <= 0 or K < 0 or not (0 <= ALPHA <= SCALE) or D <= 0 or CC_DEN <= 0 or CC_NUM < 0:
            fail("bad header")
        V = [0] * (N + 1)
        for i in range(1, N + 1):
            V[i] = int(next(it))
            if not (1 <= V[i] <= 1000):
                fail("bad V")
        W = [0] * (N + 1)
        for i in range(1, N + 1):
            W[i] = int(next(it))
            if not (1 <= W[i] <= 1000):
                fail("bad W")
    except SystemExit:
        raise
    except Exception:
        fail("malformed input")
    return N, K, ALPHA, D, CC_NUM, CC_DEN, CENTER, V, W


GAIN_POWER = 3   # cubic saturation: gain stays high inside the elbow D, then
                  # collapses fast beyond it -- a sharper elbow than 1/(1+d/D)
                  # so ordinary drift is nearly harmless but a sustained,
                  # planted extreme run truly saturates the palate.


def simulate(seq, N, K, ALPHA, D, CC_NUM, CC_DEN, CENTER, V, W):
    a = CENTER * SCALE
    prevV = CENTER
    total_err = 0
    Dp = D ** GAIN_POWER
    for tok in seq:
        if tok == 0:
            a = CENTER * SCALE
            prevV = CENTER
            continue
        Vt = V[tok]
        d = abs(a - CENTER * SCALE)
        dp = d ** GAIN_POWER
        gain = (SCALE * Dp) // (Dp + dp)            # in [0, SCALE]
        base_p = CENTER * SCALE + (gain * (Vt * SCALE - CENTER * SCALE)) // SCALE
        shift = (CC_NUM * (Vt - prevV) * SCALE) // CC_DEN
        p = base_p + shift
        err = abs(p - Vt * SCALE)
        total_err += W[tok] * err
        a = ((SCALE - ALPHA) * a + ALPHA * Vt * SCALE) // SCALE
        prevV = Vt
    return total_err


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    N, K, ALPHA, D, CC_NUM, CC_DEN, CENTER, V, W = read_instance(sys.argv[1])

    try:
        data = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    if len(data) == 0:
        fail("empty output")
    if len(data) > N + K + 5:
        fail("too many tokens")

    seq = []
    seen = [False] * (N + 1)
    n_zero = 0
    try:
        for tok_s in data:
            t = int(tok_s)
            if t == 0:
                n_zero += 1
                seq.append(0)
            elif 1 <= t <= N:
                if seen[t]:
                    fail("duplicate sample index")
                seen[t] = True
                seq.append(t)
            else:
                fail("sample index out of range")
    except SystemExit:
        raise
    except Exception:
        fail("non-integer token")

    if not all(seen[1:N + 1]):
        fail("missing sample index")
    if n_zero > K:
        fail("cleanser budget exceeded")

    F_err = simulate(seq, N, K, ALPHA, D, CC_NUM, CC_DEN, CENTER, V, W)
    if F_err < 0:
        fail("internal error")

    baseline_seq = list(range(1, N + 1))   # identity order, no cleansers
    E_b = simulate(baseline_seq, N, K, ALPHA, D, CC_NUM, CC_DEN, CENTER, V, W)
    E_b = max(1, E_b)

    margin = max(1, E_b // 10)
    BOUND = E_b + margin
    F = BOUND - F_err
    B = margin

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = max(0.0, sc / 1000.0)
    print("F_err=%d E_b=%d Ratio: %.6f" % (F_err, E_b, ratio))


if __name__ == "__main__":
    main()
