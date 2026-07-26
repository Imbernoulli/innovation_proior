# TIER: strong
"""Feasibility-first, then optimize.  Product-of-scenario-yield scoring
means ONE scenario dipping a field below its stress threshold collapses
the whole product -- so before chasing extra yield we must guarantee every
field's soil-moisture trajectory stays >= threshold in EVERY scenario
simultaneously (the same irrigation schedule is evaluated against all K
weather books).  Phase 1 repeatedly resimulates all (field, scenario)
trajectories, finds the worst current stress violation, and patches it by
placing water exactly in the pinching week (where the drought scenario's
soil-moisture envelope actually dips) rather than spreading it evenly or
trusting the cross-scenario average.  Phase 2 spends any leftover budget on
the fields/weeks with the best marginal growth rate."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0

    def nxt():
        nonlocal idx
        v = data[idx]
        idx += 1
        return v

    T = int(nxt()); F = int(nxt()); K = int(nxt())
    total_budget = int(nxt()); weekly_cap = int(nxt())
    fields = []
    for _ in range(F):
        m0 = int(nxt()); cap = int(nxt()); cons = int(nxt())
        thresh = int(nxt()); rate = int(nxt())
        fields.append((m0, cap, cons, thresh, rate))
    rain = []
    for _ in range(K):
        rain.append([int(nxt()) for _ in range(T)])

    X = [[0] * T for _ in range(F)]
    remaining_total = total_budget
    remaining_weekly = [weekly_cap] * T

    def violations():
        """All (t, deficit, f) where field f dips below threshold at week t
        under SOME scenario, given the CURRENT irrigation plan X (deficit is
        the worst -- i.e. largest -- shortfall across scenarios at that
        (f, t), since a single schedule must clear every scenario at once)."""
        vs = []
        for f in range(F):
            m0, cap, cons, thresh, rate = fields[f]
            Xf = X[f]
            worst_per_week = [0] * T
            for k in range(K):
                rk = rain[k]
                m = m0
                for t in range(T):
                    m = m + Xf[t] + rk[t] - cons
                    if m < 0:
                        m = 0
                    elif m > cap:
                        m = cap
                    if m < thresh:
                        d = thresh - m
                        if d > worst_per_week[t]:
                            worst_per_week[t] = d
            for t in range(T):
                if worst_per_week[t] > 0:
                    vs.append((t, worst_per_week[t], f))
        return vs

    # Fix the EARLIEST pinch point first: because moisture carries forward
    # through the recurrence, watering the first week a trajectory would
    # dip below threshold also lifts (or entirely clears) every later week
    # that same shortfall would otherwise have cascaded into. Patching the
    # numerically-worst (i.e. latest, most-decayed) deficit first is a trap:
    # it burns the whole budget on one-off late-season patches with no
    # forward carryover benefit, and starves everything before it.
    max_iters = 6 * F * K + 30
    for _ in range(max_iters):
        if remaining_total <= 0:
            break
        vs = violations()
        if not vs:
            break
        vs.sort(key=lambda x: (x[0], -x[1]))
        patched = False
        for t, deficit, f in vs:
            room = min(deficit, remaining_weekly[t], remaining_total)
            if room > 0:
                X[f][t] += room
                remaining_weekly[t] -= room
                remaining_total -= room
                patched = True
                break
        if not patched:
            break  # every currently-violated week is fully capped out

    # Phase 2: every field is already feasible everywhere -- the per-week
    # bonus for extra moisture is CAPPED (thresh//2 margin) but scales with
    # rate[f], so a unit of leftover water is worth more on a high-rate
    # field. Spread leftover round-robin across every (field, week) slot,
    # weighted by rate[f] (a slot for a rate-5 field is visited ~5x as
    # often as a rate-1 field's), so higher-value fields get proportionally
    # more of the leftover without fully starving the rest.
    if remaining_total > 0:
        slots = [(f, t) for t in range(T) for f in range(F)
                 for _ in range(max(1, fields[f][4]))]
        n = len(slots)
        i = 0
        stale_passes = 0
        steps = 0
        max_steps = remaining_total + 4 * n + 50
        while remaining_total > 0 and steps < max_steps and stale_passes <= 1:
            f, t = slots[i % n]
            if remaining_weekly[t] > 0:
                X[f][t] += 1
                remaining_weekly[t] -= 1
                remaining_total -= 1
                stale_passes = 0
            elif i % n == n - 1:
                stale_passes += 1
            i += 1
            steps += 1

    print(F, T)
    for f in range(F):
        print(" ".join(str(x) for x in X[f]))


if __name__ == "__main__":
    main()
