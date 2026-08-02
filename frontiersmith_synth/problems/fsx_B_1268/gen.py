import sys, random
import model

# testId -> (K tiers, trap flag). Trap cases have strongly risk-correlated departure
# thresholds (adverse selection bites hard); mild cases have near-uniform thresholds
# (departure roughly risk-blind, so cost-plus pricing is nearly harmless).
SCHEDULE = {
    1: (3, False),
    2: (3, False),
    3: (4, True),
    4: (4, False),
    5: (5, True),
    6: (5, False),
    7: (5, True),
    8: (6, True),
    9: (6, False),
    10: (6, True),
}


def gen_bucket_dist(rng, base):
    m = rng.randint(2, 3)
    vs = []
    for k in range(m):
        spread = rng.randint(-10, 10) * (k + 1)
        vs.append(max(0, base + spread))
    # probabilities summing to 1000, each part >= 60 (avoid degenerate near-zero mass)
    if m == 2:
        p1 = rng.randint(300, 700)
        prs = [p1, 1000 - p1]
    else:
        a = rng.randint(200, 500)
        b = rng.randint(200, 1000 - a - 200)
        prs = [a, b, 1000 - a - b]
    return vs, prs


def build_tier(rng, trap):
    B = rng.randint(4, 5) if trap else rng.randint(3, 4)
    # a single price tier still spans a BOUNDED range of underlying risk (real
    # tiers are not homogeneous, but they are not order-of-magnitude wide
    # either) -- draw each bucket's base loss as a multiplicative offset from
    # one tier-level risk center, capped so max/min eloss stays under ~3x.
    tier_center = rng.randint(60, 220)
    raw = []
    for j in range(B):
        factor = rng.uniform(0.60, 1.70)
        base = max(5, round(tier_center * factor))
        vs, prs = gen_bucket_dist(rng, base)
        eloss = sum(v * p for v, p in zip(vs, prs)) / 1000.0
        raw.append({"v": vs, "pr": prs, "eloss": eloss})
    # order buckets by risk ascending, but give lower-risk buckets larger current
    # market share (realistic: most policyholders are low-risk) -- this also means
    # the population that CAN leave (cheap risks) is the bulk of current volume.
    raw.sort(key=lambda x: x["eloss"])
    Bn = len(raw)
    for rank, bkt in enumerate(raw):
        base_n = rng.randint(700, 1600)
        decay = rank * rng.randint(90, 220)
        bkt["n"] = max(90, base_n - decay)

    fair = sum(b["n"] * b["eloss"] for b in raw) / max(1, sum(b["n"] for b in raw))

    # competitor undercuts the blended fair value somewhat
    c = max(1, round(fair * rng.uniform(0.85, 0.98)))

    # last cycle's price: a small, near-breakeven markup over the blended fair
    # value -- last cycle was priced conservatively, so today's baseline should
    # carry only a thin (but positive) margin and negligible departure.
    p0 = max(c + 1, round(fair * rng.uniform(1.01, 1.06)))

    cap = rng.randint(15, 35)

    # the naive cost-plus target an "obvious" solver would charge THIS cycle
    # (blended pool average + flat markup) -- used only to anchor the elasticity
    # thresholds at a realistic gap scale; greedy.py recomputes this itself from
    # the input so there is no leakage of hidden state.
    greedy_target = round(fair * 1.15)
    gap0 = max(1, p0 - c)
    gap_ref = max(gap0 * 2, greedy_target - c, 10)

    for rank, bkt in enumerate(raw):
        f = rank / (Bn - 1) if Bn > 1 else 0.0
        if trap:
            # strong risk-departure correlation: cheap/low-risk buckets start
            # leaving just past the BASELINE gap and ramp out over roughly the
            # cost-plus target's gap; the priciest bucket's threshold sits well
            # beyond that gap, so it barely budges at the cost-plus price.
            tlo = round(gap0 * 1.3 + f * gap_ref * 0.7)
            thi = tlo + round(gap_ref * (0.5 - 0.1 * f))
        else:
            # near risk-blind elasticity: everyone leaves at roughly the same
            # gap regardless of risk, so adverse selection barely matters.
            tlo = round(gap0 * 1.3 + f * gap_ref * 0.05)
            thi = tlo + round(gap_ref * 0.55)
        thi = max(thi, tlo + 1)
        bkt["tlo"] = tlo
        bkt["thi"] = thi

    buckets = [{"n": b["n"], "v": b["v"], "pr": b["pr"], "tlo": b["tlo"], "thi": b["thi"]} for b in raw]
    return {"p0": p0, "c": c, "cap": cap, "buckets": buckets}


def main():
    testId = int(sys.argv[1])
    K, trap = SCHEDULE.get(testId, (4, testId % 2 == 0))
    rng = random.Random(20260726 + 97 * testId)

    tiers = []
    for _ in range(K):
        tiers.append(build_tier(rng, trap))

    out = [str(K)]
    for t in tiers:
        out.append(f"{t['p0']} {t['c']} {t['cap']} {len(t['buckets'])}")
        for b in t["buckets"]:
            out.append(f"{b['n']} {len(b['v'])}")
            pairs = " ".join(f"{v} {p}" for v, p in zip(b["v"], b["pr"]))
            out.append(pairs)
            out.append(f"{b['tlo']} {b['thi']}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
