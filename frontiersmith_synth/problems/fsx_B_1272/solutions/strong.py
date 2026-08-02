# TIER: strong
# Constrained allocation: first GUARANTEE every typology's mandatory coverage floor using
# that typology's OWN best score/cost-density alerts (so the regulator-mandated typology is
# never skipped just because its typology-wide score level looks unpromising next to
# others, and the mandate is met economically instead of by overspending on the single
# highest-score alert regardless of price), then spend whatever time budget remains by
# global score/cost density across every typology. Two genuinely different phases -- not
# "greedy plus more iterations".
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    test_id = int(next(it)); N = int(next(it)); K = int(next(it)); C = int(next(it))
    alerts = {}
    for _ in range(N):
        aid = int(next(it)); t = int(next(it)); cost = int(next(it)); score = int(next(it))
        alerts[aid] = (t, cost, score)
    min_cover = {}
    for _ in range(K):
        t = int(next(it)); mc = int(next(it))
        min_cover[t] = mc

    by_t = {}
    for aid, (t, cost, score) in alerts.items():
        by_t.setdefault(t, []).append(aid)

    # Phase 1: satisfy every floor with the typology's OWN best score/cost-density alerts
    # (meets the mandate economically, not by overspending on the priciest high scorer).
    sel = set()
    cost_used = 0
    for t in range(1, K + 1):
        need = min_cover.get(t, 0)
        if need <= 0:
            continue
        cand = sorted(by_t.get(t, []),
                      key=lambda a: (-(alerts[a][2] / alerts[a][1]), alerts[a][1], a))
        for aid in cand[:need]:
            c = alerts[aid][1]
            if cost_used + c <= C:  # capacity guaranteed by generator, but stay defensive
                sel.add(aid); cost_used += c

    # Phase 2: spend remaining budget by global score/cost density, across all typologies.
    remaining = [a for a in alerts if a not in sel]
    remaining.sort(key=lambda a: (-(alerts[a][2] / alerts[a][1]), alerts[a][1], a))
    for aid in remaining:
        c = alerts[aid][1]
        if cost_used + c <= C:
            sel.add(aid); cost_used += c

    out = [str(len(sel))]
    out.extend(str(a) for a in sorted(sel))
    print("\n".join(out))


if __name__ == "__main__":
    main()
