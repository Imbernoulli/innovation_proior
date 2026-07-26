#!/usr/bin/env python3
"""
gen.py <testId> -- emits ONE irrigation-planning instance to stdout.

Instance shape (all integers):
  T F K
  TotalBudget WeeklyCap
  F lines: m0 cap cons thresh rate           (one crop/field per line)
  K lines: T rainfall integers                (one weather scenario per line)

Design: a season of T weeks, F fields (crops) with distinct soil-moisture
recurrences, K weather scenarios sharing one weekly rainfall stream but
diverging sharply during a "drought window" planted in >=3 tests.  Within
that window the DROUGHT scenario's rain crashes to ~0 while every OTHER
scenario's rain is deliberately boosted well above baseline, so the
*average* rainfall across scenarios during the window looks generous even
though one scenario is starving -- the trap for an average-based planner.
"""
import random
import sys

# (T, F, K, plant_drought)
TABLE = {
    1:  (8,  3, 3, False),
    2:  (10, 3, 4, False),
    3:  (10, 4, 4, False),
    4:  (12, 4, 5, True),
    5:  (12, 5, 5, False),
    6:  (14, 5, 5, False),
    7:  (14, 6, 6, True),
    8:  (16, 6, 6, False),
    9:  (18, 7, 7, False),
    10: (20, 8, 8, True),
}


def gen(test_id: int):
    T, F, K, drought = TABLE[test_id]
    rng = random.Random(20000 + 97 * test_id)

    fields = []
    for idx in range(F):
        # Field 0 is deliberately "the thirstiest crop": a bit thirstier,
        # a thinner buffer, and the highest yield rate, so it both
        # dominates the scenario sum AND is the one most exposed to a
        # masked dry spell -- "one drought scenario crosses a stress
        # threshold on the thirstiest crop" (the seed's own trap
        # language). The gap to the other fields is deliberately mild so
        # the flat baseline still handles an ordinary (non-drought) season
        # fine -- only a genuine multi-week dry spell decides it.
        if idx == 0:
            cons = rng.randint(12, 14)
            buf = rng.randint(2, 3)
            rate = rng.randint(200, 240)
        else:
            cons = rng.randint(8, 10)
            buf = rng.randint(10, 16)
            rate = rng.randint(1, 2)
        cap = rng.randint(80, 160)
        thresh_hi = min(70, cap - 20)
        thresh_lo = 30
        if thresh_hi < thresh_lo:
            thresh_hi = thresh_lo
        thresh = rng.randint(thresh_lo, thresh_hi)
        m0 = thresh + buf
        fields.append((m0, cap, cons, thresh, rate))

    baseline_lo, baseline_hi = 6, 14
    rain = [[0] * T for _ in range(K)]

    drought_k = K - 1  # designated drought scenario (deterministic index)
    span = max(7, min(13, T // 3 + 5))
    if drought:
        d_start = max(1, T // 3)
        d_len = span
        d_end = min(T, d_start + d_len)
    else:
        d_start = d_end = -1

    for t in range(T):
        in_window = drought and (d_start <= t < d_end)
        for k in range(K):
            if in_window:
                if k == drought_k:
                    rain[k][t] = rng.randint(0, 1)
                else:
                    # boosted: masks the drought scenario inside the cross-scenario
                    # average, so an average-based planner sees a *wet* week here.
                    rain[k][t] = rng.randint(28, 36)
            else:
                rain[k][t] = rng.randint(baseline_lo, baseline_hi)

    sum_cons = sum(f[2] for f in fields)
    # non-drought seasons still need a budget that scales with T (a season
    # twice as long has roughly twice as many weeks of ordinary variance to
    # cover), or the checker's own flat per-slot baseline collapses toward
    # zero for large T and trivially saturates every solution's score.
    need_span = (d_end - d_start) if drought else max(4, T // 4)
    # worst-case season need: what it costs to keep every field above its own
    # stress line through a `need_span`-week dry stretch, net of its starting
    # buffer -- this is what a feasibility-first planner must exactly afford.
    worst_case_need = sum(max(0, need_span * cons - (m0 - thresh))
                           for (m0, cap, cons, thresh, rate) in fields)
    total_budget = int(round(1.2 * worst_case_need)) + 5
    total_budget = max(total_budget, sum_cons * 3)
    # floor: the checker's own baseline splits TotalBudget flat over F*T
    # slots. Field 0's consumption runs above the baseline rainfall mean
    # (~10) BY DESIGN (it is "the thirstiest crop"), so unless the flat
    # per-slot share at least covers that average gap, the baseline decays
    # even in an ordinary (non-drought) season purely from field 0's
    # structural deficit -- collapsing the baseline and trivially
    # saturating every real solution's score, not because of the planted
    # drought trap but because of an unrelated sizing artifact.
    dominant_cons = fields[0][2]
    floor_per_slot = max(2, dominant_cons - 9)
    total_budget = max(total_budget, floor_per_slot * F * T)
    weekly_cap = int(round(total_budget / max(3, need_span - 1) * 1.35))
    weekly_cap = max(weekly_cap, (sum_cons // max(1, F)) + 4)

    out = [f"{T} {F} {K}", f"{total_budget} {weekly_cap}"]
    for (m0, cap, cons, thresh, rate) in fields:
        out.append(f"{m0} {cap} {cons} {thresh} {rate}")
    for k in range(K):
        out.append(" ".join(str(x) for x in rain[k]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    tid = int(sys.argv[1])
    gen(tid)
