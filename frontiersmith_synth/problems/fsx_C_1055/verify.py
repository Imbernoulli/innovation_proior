import sys, math


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def landing_col(h, a, R, L, M):
    """Shadowing capture rule: the aim column `a` catches its own particle
    UNLESS some other column within the capture radius R is more than M
    units taller than `a` -- a genuinely tall neighbor casts a capturing
    shadow; a merely slightly-taller one does not. Among columns that clear
    the margin, the tallest wins; ties break toward the column nearest the
    aim, then toward the smaller index."""
    lo = a - R
    hi = a + R
    if lo < 0:
        lo = 0
    if hi > L - 1:
        hi = L - 1
    ha = h[a]
    best_c = None
    best_key = None
    for c in range(lo, hi + 1):
        if c == a:
            continue
        if h[c] > ha + M:
            key = (h[c], -abs(c - a), -c)
            if best_key is None or key > best_key:
                best_key = key
                best_c = c
    return best_c if best_c is not None else a


def simulate(L, R, M, schedule):
    """Deterministic ballistic deposition with shadowing capture + lateral
    sticking. `schedule` is a list of aim columns, one per time step.
    Returns the final height array."""
    h = [0] * L
    for a in schedule:
        c = landing_col(h, a, R, L, M)
        left = h[c - 1] if c - 1 >= 0 else -1
        right = h[c + 1] if c + 1 < L else -1
        h[c] = max(h[c] + 1, left, right)
    return h


def sq_err(h, target):
    return float(sum((hv - tv) * (hv - tv) for hv, tv in zip(h, target)))


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")
    try:
        out_toks = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    try:
        it = iter(inp)
        L = int(next(it)); R = int(next(it)); M = int(next(it)); T = int(next(it))
        if L <= 0 or R < 0 or M < 0 or T <= 0:
            fail("bad header")
        target = [int(next(it)) for _ in range(L)]
        if any(v < 0 for v in target):
            fail("bad target")
    except Exception:
        fail("bad input")

    # ---- parse participant output: exactly T integer aim columns ----
    if len(out_toks) != T:
        fail("expected exactly %d tokens, got %d" % (T, len(out_toks)))
    schedule = []
    try:
        for tok in out_toks:
            v = float(tok)
            if not math.isfinite(v):
                fail("non-finite token")
            iv = int(tok)
            if iv != v:
                fail("non-integer token")
            if iv < 0 or iv >= L:
                fail("aim column out of range")
            schedule.append(iv)
    except SystemExit:
        raise
    except Exception:
        fail("malformed schedule")

    # ---- internal baseline B: round-robin flux, oblivious to the target
    # and to the shadowing/capture dynamics -- a simple, positive, trivial
    # construction. ----
    baseline_schedule = [t % L for t in range(T)]
    h_base = simulate(L, R, M, baseline_schedule)
    B = sq_err(h_base, target)
    B = max(B, 1e-6)

    h_final = simulate(L, R, M, schedule)
    F = sq_err(h_final, target)
    F = max(F, 0.0)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.4f B=%.4f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
