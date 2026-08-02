import sys, math

K = 5
# must mirror gen.py's TRAP_IDS exactly: rounds where the regulator-mandated typology
# carries a binding coverage floor AND the hard-to-score/high-value hidden structure.
TRAP_IDS = {3, 5, 8, 10}


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def reg_typology(test_id):
    return ((test_id - 1) % K) + 1


def h(*args):
    """Deterministic 64-bit integer hash -> float in [0,1). Pure integer arithmetic only,
    so it reproduces bit-for-bit on any machine (no libm, no random module)."""
    x = 0x9E3779B97F4A7C15
    for a in args:
        x = (x ^ (a & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
        x = (x * 1000003 + 0xABCDEF1234567) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 33)
        x = (x * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 33)
        x = (x * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 33)
    return (x % 1_000_000) / 1_000_000.0


def hidden_params(test_id, R, t):
    """Hidden (checker-only) precision/value structure for typology t this round.
    On a trap round, R is scored LOW by the visible model score but carries a non-trivial
    base precision AND the largest value-per-hit -- a solver reading only the visible score
    systematically under-estimates it."""
    if test_id in TRAP_IDS and t == R:
        base_precision, slope, base_value = 0.03, 1.30, 280.0
    else:
        base_precision, slope, base_value = 0.02, 0.90, 70.0 + 20.0 * ((t + 2) % 4)
    return base_precision, slope, base_value


def recovered_value(test_id, R, aid, t, score):
    """Expected recovered value if this alert is worked: precision(score,typology) x
    value-per-confirmed-hit x a small per-alert jitter. Deliberately NOT a Bernoulli
    hit/miss draw -- with only tens of alerts per typology per case, a coin-flip outcome
    would make the objective dominated by small-sample luck instead of by the quality of
    the chosen allocation. This keeps the reward a smooth, allocation-quality signal while
    still hiding the exact precision/value structure from the solver."""
    base_precision, slope, base_value = hidden_params(test_id, R, t)
    p = base_precision + slope * (score / 100.0)
    p = max(0.0, min(0.97, p))
    jitter = 0.85 + 0.3 * h(test_id, aid, t, 4)
    return p * base_value * jitter


def main():
    inp = open(sys.argv[1]).read().split()
    out_txt = open(sys.argv[2]).read()

    try:
        it = iter(inp)
        test_id = int(next(it))
        N = int(next(it))
        Kfile = int(next(it))
        C = int(next(it))
        if Kfile != K:
            fail("typology count mismatch")
        alerts = {}
        typology_of = {}
        cost_of = {}
        score_of = {}
        for _ in range(N):
            aid = int(next(it)); t = int(next(it)); cost = int(next(it)); score = int(next(it))
            alerts[aid] = (t, cost, score)
            typology_of[aid] = t; cost_of[aid] = cost; score_of[aid] = score
        min_cover = {}
        for _ in range(K):
            t = int(next(it)); mc = int(next(it))
            min_cover[t] = mc
    except Exception:
        fail("bad input")

    R = reg_typology(test_id)

    # ---- internal baseline B: coverage-first cheapest fill, then arbitrary ascending fill ----
    by_t = {t: [aid for aid in alerts if alerts[aid][0] == t] for t in range(1, K + 1)}
    for t in by_t:
        by_t[t].sort(key=lambda a: (cost_of[a], a))

    base_sel = set()
    base_cost = 0
    for t in range(1, K + 1):
        need = min_cover.get(t, 0)
        for aid in by_t[t][:need]:
            base_sel.add(aid); base_cost += cost_of[aid]
    for aid in sorted(alerts):
        if aid in base_sel:
            continue
        c = cost_of[aid]
        if base_cost + c <= C:
            base_sel.add(aid); base_cost += c

    B = sum(recovered_value(test_id, R, aid, typology_of[aid], score_of[aid]) for aid in base_sel)
    B = max(B, 1.0)

    # ---- parse participant output: k, then k alert ids ----
    toks = out_txt.split()
    try:
        k = int(toks[0])
    except Exception:
        fail("parse")
    if k < 0 or k > N:
        fail("k out of range")
    ids_tok = toks[1:1 + k]
    if len(ids_tok) != k:
        fail("truncated output")
    try:
        ids = [int(x) for x in ids_tok]
    except Exception:
        fail("non-integer id")
    for v in ids:
        if v != v or math.isinf(v):
            fail("non-finite id")

    seen = set()
    total_cost = 0
    per_t_count = {t: 0 for t in range(1, K + 1)}
    for aid in ids:
        if aid < 1 or aid > N or aid not in alerts:
            fail("id out of range %d" % aid)
        if aid in seen:
            fail("duplicate id %d" % aid)
        seen.add(aid)
        t, cost, score = alerts[aid]
        total_cost += cost
        per_t_count[t] += 1

    if total_cost > C:
        fail("capacity exceeded %d > %d" % (total_cost, C))
    for t in range(1, K + 1):
        need = min_cover.get(t, 0)
        if per_t_count[t] < need:
            fail("coverage floor violated typology %d: %d < %d" % (t, per_t_count[t], need))

    F = sum(recovered_value(test_id, R, aid, typology_of[aid], score_of[aid]) for aid in ids)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.3f B=%.3f k=%d Ratio: %.6f" % (F, B, k, sc / 1000.0))


if __name__ == "__main__":
    main()
