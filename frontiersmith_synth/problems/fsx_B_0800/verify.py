import sys, math, bisect


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("bad input file")
    try:
        it = iter(inp)
        T = int(next(it))
        K, A, Acol, gamma, p_base, Qcap = (float(next(it)) for _ in range(6))
        S0, r_base = (float(next(it)) for _ in range(2))
        dstart = int(next(it)); dend = int(next(it)); drought_mult = float(next(it))
        n_closed = int(next(it))
        closed_mod = set(int(next(it)) for _ in range(n_closed))
        n_season = int(next(it))
        price_season = [float(next(it)) for _ in range(n_season)]
        N = int(next(it))
        costs = []
        caps = []
        for _ in range(N):
            c = float(next(it)); cap = float(next(it))
            costs.append(c); caps.append(cap)
    except Exception:
        fail("bad input parse")

    # ---- parse participant output: T finite floats ----
    try:
        raw = open(sys.argv[2]).read().split()
    except Exception:
        fail("bad output file")
    if len(raw) < T:
        fail("too few quotas: got %d need %d" % (len(raw), T))
    try:
        quotas = [float(x) for x in raw[:T]]
    except Exception:
        fail("non-numeric quota")
    for q in quotas:
        if not math.isfinite(q):
            fail("non-finite quota")
        if q < -1e-9 or q > Qcap + 1e-6:
            fail("quota out of range [0,%.3f]" % Qcap)

    # ---- sort fishers by cost once, prefix-sum caps for O(log N) effort-race lookups ----
    order = sorted(range(N), key=lambda idx: costs[idx])
    sorted_costs = [costs[idx] for idx in order]
    prefix = [0.0]
    for idx in order:
        prefix.append(prefix[-1] + caps[idx])

    def attempted_catch(thresh):
        j = bisect.bisect_left(sorted_costs, thresh)
        return prefix[j]

    # ---- replay the recurrence: depensation growth -> effort race -> harvest ----
    S = S0
    total = 0.0
    disc = 1.0
    for t in range(T):
        wk = t % 52
        Q = max(0.0, quotas[t])
        if wk in closed_mod and Q > 1e-6:
            fail("nonzero quota %.4f in legally closed week (t=%d)" % (Q, t))

        r_t = r_base * (drought_mult if dstart <= t < dend else 1.0)
        growth = r_t * S * (S / A - 1.0) * (1.0 - S / K)
        Sp = S + growth
        if Sp < 0.0:
            Sp = 0.0

        p_t = p_base * price_season[wk]
        thresh = p_t * (Sp / K)
        attempt = attempted_catch(thresh)
        H = min(Q, attempt, Sp)

        Snext = Sp - H
        if Snext < Acol - 1e-6:
            fail("stock collapse at week %d: S=%.3f < Acol=%.3f" % (t, Snext, Acol))

        total += disc * p_t * H
        disc *= gamma
        S = Snext

    F = total

    # ---- internal baseline B: flat conservative constant quota, 1.2% of S0, ----
    # ---- respecting only the mandatory legal closed weeks (naive, non-adaptive) ----
    Qb = min(0.012 * S0, Qcap)
    Sb = S0
    Btot = 0.0
    discb = 1.0
    collapsed = False
    for t in range(T):
        wk = t % 52
        Qh = 0.0 if wk in closed_mod else Qb
        r_t = r_base * (drought_mult if dstart <= t < dend else 1.0)
        growth = r_t * Sb * (Sb / A - 1.0) * (1.0 - Sb / K)
        Spb = Sb + growth
        if Spb < 0.0:
            Spb = 0.0
        p_t = p_base * price_season[wk]
        thresh = p_t * (Spb / K)
        attempt = attempted_catch(thresh)
        Hb = min(Qh, attempt, Spb)
        Sbn = Spb - Hb
        if Sbn < Acol - 1e-6:
            collapsed = True
            break
        Btot += discb * p_t * Hb
        discb *= gamma
        Sb = Sbn
    B = max(1e-9, Btot) if not collapsed else 1e-9

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.3f B=%.3f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
