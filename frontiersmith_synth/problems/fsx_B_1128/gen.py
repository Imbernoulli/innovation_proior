#!/usr/bin/env python3
"""
gen.py <testId> -- emits ONE sourdough-bakery vitality-phase-alignment instance.

Deterministic: all randomness is seeded purely from testId.

Input format printed to stdout:
  line 1:            T H M BUDGET
  next T lines:       GROW BASE_KICK DECAY SAT CRASH_DIV COST     (one per tank, index 0..T-1)
  next M lines:       DAY THETA VALUE                              (one per order, index 0..M-1)

Model (see statement.md for the full narrative):
  vitality v in [0,SCALE], satiety counter n.
  fed day:   n += 1; if n > SAT: crash (v //= CRASH_DIV, n = 0)
                      else: v += floor(GROW*v*(SCALE-v)/SCALE^2) + BASE_KICK  (logistic kick), capped at SCALE
  unfed day: n = max(0, n-1); v = floor(v*(1000-DECAY)/1000)                  (decay)
  Order j is fulfilled iff some tank's vitality AT DAY_j (i.e. after DAY_j steps from t=0,
  v=0,n=0) is >= THETA_j.
"""
import random
import sys

SCALE = 200


def transition(v, n, fed, GROW, BASE_KICK, DECAY, SAT, CRASH_DIV):
    if fed:
        n2 = n + 1
        if n2 > SAT:
            v2 = v // CRASH_DIV
            n2 = 0
        else:
            growth = (GROW * v * (SCALE - v)) // (SCALE * SCALE)
            v2 = v + growth + BASE_KICK
            if v2 > SCALE:
                v2 = SCALE
    else:
        n2 = n - 1
        if n2 < 0:
            n2 = 0
        v2 = (v * (1000 - DECAY)) // 1000
    return v2, n2


def cont_array(H, params):
    """Vitality after e consecutive feed days starting fresh (v=0,n=0) at t=0, e=0..H."""
    GROW, BASE_KICK, DECAY, SAT, CRASH_DIV = params
    v, n = 0, 0
    arr = [0]
    for _ in range(H):
        v, n = transition(v, n, True, GROW, BASE_KICK, DECAY, SAT, CRASH_DIV)
        arr.append(v)
    return arr


INF = float("inf")


def value_covered(cont_i, s, H, orders, covered):
    """Orders newly satisfied if tank i is fed continuously from day s to H-1
    (so its vitality at day s+e is cont_i[e]).  `covered` marks orders already
    claimed by an earlier pick (so callers can run a set-cover greedy)."""
    total = 0
    newly = []
    for j, (day, theta, val) in enumerate(orders):
        if covered[j] or day < s:
            continue
        e = day - s
        if e < len(cont_i) and cont_i[e] >= theta:
            total += val
            newly.append(j)
    return total, newly


def budgeted_coverage(conts, costs, orders, H, budget, s_values, tank_order):
    """Greedy budgeted-maximum-coverage: repeatedly activate the (tank, start
    day) with the best newly-covered-value / flour-cost ratio, feeding that
    tank continuously from `s` through day H-1. Each tank is used at most
    once. Returns (total_value, schedule=[(tank,s),...])."""
    T = len(conts)
    M = len(orders)
    covered = [False] * M
    used = [False] * T
    remaining = budget
    schedule = []
    total_val = 0
    while True:
        best = None
        for i in tank_order:
            if used[i]:
                continue
            cont_i = conts[i]
            cost_i = costs[i]
            for s in s_values:
                cost = (H - s) * cost_i
                if cost <= 0 or cost > remaining:
                    continue
                val, newly = value_covered(cont_i, s, H, orders, covered)
                if val <= 0:
                    continue
                ratio = val / cost
                if best is None or ratio > best[0] + 1e-12:
                    best = (ratio, i, s, cost, val, newly)
        if best is None:
            break
        _, i, s, cost, val, newly = best
        used[i] = True
        remaining -= cost
        for j in newly:
            covered[j] = True
        schedule.append((i, s))
        total_val += val
    return total_val, schedule


def best_of_coverage(conts, costs, orders, H, budget):
    """Ensemble: try full phase search plus a couple of deterministic
    tie-break variants, keep whichever total value is largest. Always
    dominates the s=0-only ('start immediately') strategy since that is one
    of the candidates."""
    T = len(conts)
    candidates = [
        budgeted_coverage(conts, costs, orders, H, budget, range(H), range(T)),
        budgeted_coverage(conts, costs, orders, H, budget, (0,), range(T)),
        budgeted_coverage(conts, costs, orders, H, budget, range(H),
                           sorted(range(T), key=lambda i: (-costs[i], i))),
        budgeted_coverage(conts, costs, orders, H, budget, range(H),
                           sorted(range(T), key=lambda i: (costs[i], i))),
    ]
    return max(candidates, key=lambda c: c[0])


def _ramp_days(GROW, BASE_KICK, target=150, cap=200):
    """Days of uninterrupted feeding a fresh tank needs to first reach `target`
    (crash/decay disabled) -- used to calibrate SAT so every randomly-drawn
    tank actually gets a real peak before it is allowed to crash."""
    v, n = 0, 0
    for t in range(1, cap):
        v, n = transition(v, n, True, GROW, BASE_KICK, 10 ** 9, 10 ** 9, 10 ** 9)
        if v >= target:
            return t
    return cap


def gen_tanks(rng, T, uniform):
    # Tuned so vitality is a genuine narrow PULSE (a steep logistic ramp
    # taking 1-2 real cycles, a handful of days near cap, then a crash) not
    # a wide plateau -- a continuously-fed tank sits high only ~15% of the
    # time, so two independently-dynamicked tanks are rarely both near peak
    # on the same calendar day. That scarcity is what makes timing matter.
    # SAT is derived from the tank's own ramp speed (not drawn independently)
    # so slow-growing tanks still get to peak before their first crash.
    if uniform:
        GROW = rng.randint(95, 140)
        BASE_KICK = rng.randint(1, 3)
        DECAY = rng.randint(16, 24)
        SAT = _ramp_days(GROW, BASE_KICK) + rng.randint(1, 3)
        CRASH_DIV = rng.randint(13, 18)
        params = [(GROW, BASE_KICK, DECAY, SAT, CRASH_DIV)] * T
    else:
        params = []
        for _ in range(T):
            GROW = rng.randint(85, 150)
            BASE_KICK = rng.randint(1, 3)
            DECAY = rng.randint(14, 26)
            SAT = _ramp_days(GROW, BASE_KICK) + rng.randint(1, 3)
            CRASH_DIV = rng.randint(12, 20)
            params.append((GROW, BASE_KICK, DECAY, SAT, CRASH_DIV))
    costs = [rng.randint(3, 10) for _ in range(T)]
    return params, costs


def make_instance(testId):
    rng = random.Random(1000003 * testId + 7919)

    # (T, H, M, trap?) per difficulty rung
    plan = {
        1: (2, 16, 3, False),
        2: (3, 22, 4, False),
        3: (3, 26, 4, True),
        4: (4, 30, 5, True),
        5: (4, 34, 6, True),
        6: (4, 36, 5, False),
        7: (5, 42, 7, True),
        8: (6, 46, 6, False),
        9: (7, 52, 8, True),
        10: (8, 60, 10, True),
    }
    T, H, M, trap = plan[testId]

    params, costs = gen_tanks(rng, T, uniform=trap)
    conts_full = [cont_array(H, p) for p in params]
    cheapest = min(range(T), key=lambda i: (costs[i], i))

    orders = []

    # order 0: guaranteed "easy" order the baseline (cheapest tank, fed continuously
    # the whole horizon) always fulfils -- anchors a positive, non-saturating baseline.
    day_easy = max(2, H // 6)
    theta_easy = conts_full[cheapest][day_easy]
    theta_easy = max(1, theta_easy)
    orders.append((day_easy, theta_easy, 100))

    if trap:
        arr = conts_full[0]  # all tanks share this trajectory shape when trap=True
        peak_upto = []
        best = 0
        for x in arr:
            if x > best:
                best = x
            peak_upto.append(best)
        valleys = [d for d in range(2, len(arr)) if arr[d] < arr[d - 1] * 0.6 and d >= 6]
        # skip the very first valley or two so a real burst-window exists before it
        usable = [d for d in valleys if d <= H - 1 and peak_upto[d] - arr[d] >= 40]
        n_traps = min(len(usable), max(2, M // 2))
        chosen = usable[:n_traps] if n_traps else []
        for k, d in enumerate(chosen):
            gap = peak_upto[d] - arr[d]
            theta = arr[d] + max(1, gap // 2)
            theta = min(theta, peak_upto[d])
            value = 300 + 45 * k
            orders.append((d, theta, value))

    # fill remaining slots with lighter random orders, each calibrated against
    # a tank's plain continuous-from-day-0 value at that day. The calibration
    # tank is round-robined over the NON-cheapest tanks (never the cheapest),
    # so the checker's single-cheap-tank baseline structurally cannot pick
    # these up by luck -- reaching them needs genuine flour spent on a SECOND
    # (or third) tank, exercising feed-budget-rationing beyond the planted
    # phase-alignment traps above.
    non_cheap = [i for i in range(T) if i != cheapest] or [cheapest]
    rr = 0
    while len(orders) < M:
        day = rng.randint(4, H)
        tank = non_cheap[rr % len(non_cheap)]
        rr += 1
        theta = conts_full[tank][day]
        if theta <= 1:
            continue
        value = rng.randint(60, 190)
        orders.append((day, theta, value))

    orders = orders[:M]

    unconstrained = sum(costs) * H + 10
    v_max, sched_max = best_of_coverage(conts_full, costs, orders, H, unconstrained)
    cost_at_vmax = sum((H - s) * costs[i] for (i, s) in sched_max)

    frac = 0.62 if trap else 0.85
    budget = int(cost_at_vmax * frac)
    baseline_need = H * costs[cheapest]
    costs_sorted = sorted(costs)
    if trap:
        # tanks share one dynamics shape here -- a second SAME-shape tank at
        # day 0 adds no new coverage for a non-timing solver, so a flat top-up
        # is enough (the real headroom comes from phase choice, not tank count).
        min_slack_budget = baseline_need + 8 * max(costs)
    else:
        # tanks have genuinely different dynamics -- make sure at least a
        # SECOND full-horizon tank is affordable, so an obvious recipe that
        # just throws more (different) tanks at the order book has real room.
        second_full = H * (costs_sorted[1] if len(costs_sorted) > 1 else costs_sorted[0])
        min_slack_budget = baseline_need + second_full + 4 * max(costs)
    budget = max(budget, min_slack_budget)
    budget += rng.randint(0, 3)

    return T, H, M, budget, params, costs, orders


def main():
    testId = int(sys.argv[1])
    T, H, M, BUDGET, params, costs, orders = make_instance(testId)
    out = [f"{T} {H} {M} {BUDGET}"]
    for i in range(T):
        GROW, BASE_KICK, DECAY, SAT, CRASH_DIV = params[i]
        out.append(f"{GROW} {BASE_KICK} {DECAY} {SAT} {CRASH_DIV} {costs[i]}")
    for (day, theta, value) in orders:
        out.append(f"{day} {theta} {value}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
