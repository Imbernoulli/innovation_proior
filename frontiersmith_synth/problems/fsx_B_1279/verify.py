import sys, math, random


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def parse_input(text):
    it = iter(text.split())
    try:
        T = int(next(it)); F = int(next(it))
        L = [int(next(it)) for _ in range(T)]
        y = [int(next(it)) for _ in range(T)]
        Base = [int(next(it)) for _ in range(T)]
        hn = int(next(it)); hd = int(next(it))
        S = int(next(it)); wl = int(next(it))
    except Exception:
        raise ValueError("malformed instance")
    return T, F, L, y, Base, hn, hd, S, wl


def seed_of(T, F, L, y, Base, hn, hd):
    """Deterministic seed for the HIDDEN stress scenarios, derived only from the visible
    instance (the checker is invoked as `verify.py in out ans` with no external test
    index, so the seed must come purely from the instance content)."""
    s = (T * 1000003 + F * 7919
         + sum(x * (k + 1) for k, x in enumerate(L)) * 131
         + sum(y) * 17 + sum(Base) * 13
         + hn * 97 + hd * 89)
    return s % (2 ** 31)


def build_scenarios(T, L, S, wl, seed):
    """S deterministic stress windows: the two windows centered on the largest legacy
    liability peaks are ALWAYS included first (guarantees the clustering trap really
    bites when a ladder mirrors those peaks), the remaining windows are a seeded sweep.
    None of this is printed in the visible instance -- a solver must hedge against ANY
    plausible stress date, not pattern-match one revealed scenario."""
    order = sorted(range(T), key=lambda t: -L[t])
    peaks = order[:2] if len(order) >= 2 else order
    wins = []
    for idx in peaks:
        start = max(0, min(T - wl, idx - wl // 2))
        wins.append(range(start, start + wl))
    rng = random.Random(seed + 12345)
    while len(wins) < S:
        start = rng.randint(0, T - wl)
        wins.append(range(start, start + wl))
    return wins


def objective(T, L, y, Base, hn, hd, scenarios, p):
    """cost = weighted yield cost of the ladder
             + penalty for cumulative prefunding shortfall vs. the legacy schedule
             + penalty (averaged over the hidden stress scenarios) for any date whose
               maturing face value exceeds that date's stress-haircut-adjusted rollover
               capacity."""
    cumL = 0; cumP = 0; cov_gap = 0
    for t in range(T):
        cumL += L[t]; cumP += p[t]
        if cumP < cumL:
            cov_gap += (cumL - cumP)

    yield_cost = sum(p[t] * y[t] for t in range(T)) / 100.0

    stress_total = 0
    for win in scenarios:
        wset = set(win)
        for t in range(T):
            cap = (Base[t] * hn) // hd if t in wset else Base[t]
            if p[t] > cap:
                stress_total += p[t] - cap
    stress_penalty = stress_total / len(scenarios)

    W_COV = 4.0
    W_STRESS = 18.0
    return yield_cost + W_COV * cov_gap + W_STRESS * stress_penalty


def main():
    in_text = open(sys.argv[1]).read()
    out_text = open(sys.argv[2]).read()

    try:
        T, F, L, y, Base, hn, hd, S, wl = parse_input(in_text)
    except Exception:
        fail("bad instance file")

    if T <= 0 or F < 0 or wl <= 0 or wl > T or S <= 0:
        fail("bad instance parameters")

    seed = seed_of(T, F, L, y, Base, hn, hd)
    scenarios = build_scenarios(T, L, S, wl, seed)

    # ---- checker's own always-feasible baseline B: dump the entire funding need into
    # the first maturity ("borrow it all overnight, ignore rollover risk entirely") ----
    triv_p = [0] * T
    triv_p[0] = F
    B = max(1e-9, objective(T, L, y, Base, hn, hd, scenarios, triv_p))

    # ---- parse & strictly validate the participant artifact ----
    toks = out_text.split()
    if len(toks) != T:
        fail("expected exactly %d integers, got %d" % (T, len(toks)))
    p = []
    for tok in toks:
        try:
            v = float(tok)
        except ValueError:
            fail("non-numeric token %r" % tok)
        if not math.isfinite(v):
            fail("non-finite token %r" % tok)
        if abs(v - round(v)) > 1e-6:
            fail("non-integer token %r" % tok)
        iv = int(round(v))
        if iv < 0:
            fail("negative face value at maturity")
        if iv > 10 ** 15:
            fail("out-of-range face value")
        p.append(iv)

    if sum(p) != F:
        fail("face values must sum to exactly F=%d, got %d" % (F, sum(p)))

    Fobj = objective(T, L, y, Base, hn, hd, scenarios, p)
    sc = min(1000.0, 100.0 * B / max(1e-9, Fobj))
    print("F_obj=%.4f B=%.4f Ratio: %.6f" % (Fobj, B, sc / 1000.0))


if __name__ == "__main__":
    main()
