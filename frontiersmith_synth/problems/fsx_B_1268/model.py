# Shared deterministic pricing model used by gen.py (to calibrate instances) and
# verify.py (to score). NOT visible to sandboxed solutions (by harness design) --
# solutions must reimplement whatever pieces of this they need themselves.


def parse_instance(tokens):
    """tokens: list of str/whitespace-split tokens for the whole input file.
    Returns list of tier dicts: {p0,c,cap,buckets:[{n,v[],pr[],tlo,thi}]}"""
    it = iter(tokens)

    def nxt_int():
        return int(next(it))

    K = nxt_int()
    tiers = []
    for _ in range(K):
        p0 = nxt_int(); c = nxt_int(); cap = nxt_int(); B = nxt_int()
        buckets = []
        for _ in range(B):
            n = nxt_int(); m = nxt_int()
            vs = []; prs = []
            for _ in range(m):
                vs.append(nxt_int()); prs.append(nxt_int())
            tlo = nxt_int(); thi = nxt_int()
            buckets.append({"n": n, "v": vs, "pr": prs, "tlo": tlo, "thi": thi})
        tiers.append({"p0": p0, "c": c, "cap": cap, "buckets": buckets})
    return tiers


def bucket_eloss(b):
    return sum(v * pr for v, pr in zip(b["v"], b["pr"])) / 1000.0


def depart_frac(gap, tlo, thi):
    if gap <= tlo:
        return 0.0
    if gap >= thi:
        return 1.0
    if thi <= tlo:
        return 1.0
    return (gap - tlo) / float(thi - tlo)


def tier_profit(tier, price):
    gap = price - tier["c"]
    claims = 0.0
    remain_total = 0.0
    for b in tier["buckets"]:
        frac = depart_frac(gap, b["tlo"], b["thi"])
        remaining = b["n"] * (1.0 - frac)
        claims += remaining * bucket_eloss(b)
        remain_total += remaining
    revenue = price * remain_total
    return revenue - claims


def tier_band(tier):
    p0 = tier["p0"]; cap = tier["cap"]
    delta = (p0 * cap) // 100
    lo = p0 - delta
    hi = p0 + delta
    if lo < 0:
        lo = 0
    return lo, hi


def total_profit(tiers, prices):
    return sum(tier_profit(t, p) for t, p in zip(tiers, prices))


def baseline_profit(tiers):
    return sum(tier_profit(t, t["p0"]) for t in tiers)
