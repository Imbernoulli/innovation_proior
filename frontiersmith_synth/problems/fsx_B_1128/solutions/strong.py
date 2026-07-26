# TIER: strong
"""Insight: with this crash-reset dynamic, a tank fed continuously is a PULSE
TRAIN, not a ramp to a plateau -- it climbs, crashes once satiety overflows,
climbs again, forever. Since the whole shape only depends on ELAPSED feed
days (not the calendar date you started), the only real decision per tank is
WHEN to start it: pick a start day s and feed continuously from s onward, and
the tank's entire future trajectory is just the same pulse shape shifted by
s. Choosing s well can align that pulse's peaks with SEVERAL orders at once
("phase-locking an oscillator to the calendar") -- something "start
immediately" can never do. We then run a budgeted-maximum-coverage greedy
over (tank, start day) choices: repeatedly activate whichever (tank, s) adds
the most newly-covered order value per unit flour, until the budget is gone.
Some tanks are deliberately never activated (sacrificed). We try a few
deterministic tie-break variants and keep the best, so this solution can
never do worse than the naive "everyone starts at day 0" recipe."""
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
    GROW, BASE_KICK, DECAY, SAT, CRASH_DIV = params
    v, n = 0, 0
    arr = [0]
    for _ in range(H):
        v, n = transition(v, n, True, GROW, BASE_KICK, DECAY, SAT, CRASH_DIV)
        arr.append(v)
    return arr


def value_covered(cont_i, s, H, orders, covered):
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


def main():
    tokens = sys.stdin.read().split()
    idx = 0

    def nxt(k):
        nonlocal idx
        vals = tokens[idx: idx + k]
        idx += k
        return [int(x) for x in vals]

    T, H, M, BUDGET = nxt(4)
    tanks = []
    for _ in range(T):
        tanks.append(nxt(6))  # GROW,BASE_KICK,DECAY,SAT,CRASH_DIV,COST
    orders = []
    for _ in range(M):
        day, theta, value = nxt(3)
        orders.append((day, theta, value))

    conts = [cont_array(H, t[:5]) for t in tanks]
    costs = [t[5] for t in tanks]

    candidates = [
        budgeted_coverage(conts, costs, orders, H, BUDGET, range(H), range(T)),
        budgeted_coverage(conts, costs, orders, H, BUDGET, (0,), range(T)),
        budgeted_coverage(conts, costs, orders, H, BUDGET, range(H),
                           sorted(range(T), key=lambda i: (-costs[i], i))),
        budgeted_coverage(conts, costs, orders, H, BUDGET, range(H),
                           sorted(range(T), key=lambda i: (costs[i], i))),
    ]
    _best_val, best_sched = max(candidates, key=lambda c: c[0])

    schedule = []
    for (tank, s) in best_sched:
        for d in range(s, H):
            schedule.append((tank, d))

    lines = [str(len(schedule))]
    for (i, d) in schedule:
        lines.append(f"{i} {d}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
