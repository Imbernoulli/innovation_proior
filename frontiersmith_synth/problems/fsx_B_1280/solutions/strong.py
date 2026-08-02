# TIER: strong
"""Stratified sampling design, not value-chasing.

Insight: a stratum's audited-sample finding can only be legitimately projected onto
its whole population if the sample is both REPRESENTATIVE (spans the value range,
mean value-percentile near the stratum's own center) and PRECISE (relative margin of
error, estimated from the sample's own dispersion via the historical prior stdev, is
within this test's stated threshold). Judgmental "biggest transactions" sampling gets
neither property for free.

Plan:
  1. Fully census whichever stratum is cheapest to audit completely (its true total
     becomes certain money, zero extrapolation risk -- always worth banking first).
  2. For every other stratum, size a target sample count from the PRIOR error-rate
     coefficient of variation (given in the input) via the same relative-margin
     formula the checker uses, with a small safety buffer.
  3. Fill that target with a SYSTEMATIC, value-spread pick (evenly spaced across the
     stratum's own value-sorted order) so the mean percentile lands near the stratum's
     center -- satisfying representativeness by construction -- spending whatever
     budget remains after the census.
"""
import sys, math

Z = 1.645


def systematic_pick(members_by_val, target):
    nb = len(members_by_val)
    if target <= 0 or nb == 0:
        return []
    step = nb / target
    picks = []
    for k in range(target):
        i = min(nb - 1, int(k * step + step / 2))
        picks.append(members_by_val[i])
    return picks


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    nxt = lambda: next(it)

    t = int(nxt()); N = int(nxt()); K = int(nxt()); Cmax = int(nxt())
    thresh = float(nxt())
    priors = []
    for _ in range(K):
        pm = float(nxt()); ps = float(nxt())
        priors.append((pm, ps))
    rows = []
    for _ in range(N):
        tid = int(nxt()); h = int(nxt()); v = int(nxt()); cost = int(nxt())
        rows.append((tid, h, v, cost))

    members = {h: [r for r in rows if r[1] == h] for h in range(K)}
    full_cost = {h: sum(r[3] for r in members[h]) for h in range(K)}
    h_census = min(full_cost, key=lambda h: full_cost[h])

    chosen = []
    budget = Cmax
    if full_cost[h_census] <= budget:
        chosen += [r[0] for r in members[h_census]]
        budget -= full_cost[h_census]
        censused = {h_census}
    else:
        censused = set()

    remaining = [h for h in range(K) if h not in censused]
    # spend the leftover budget on the cheapest strata first, so a modest budget still
    # buys a defensible sample somewhere rather than nothing everywhere
    remaining.sort(key=lambda h: full_cost[h])

    for h in remaining:
        pm, ps = priors[h]
        cv = ps / max(pm, 1e-9)
        n0 = math.ceil((Z * cv / max(thresh, 1e-6)) ** 2)
        target = min(len(members[h]), max(n0, 6) + 3)
        by_val = sorted(members[h], key=lambda r: (r[2], r[0]))
        for r in systematic_pick(by_val, target):
            tid, hh, v, cost = r
            if cost <= budget:
                chosen.append(tid)
                budget -= cost

    print(" ".join(map(str, chosen)))


if __name__ == "__main__":
    main()
