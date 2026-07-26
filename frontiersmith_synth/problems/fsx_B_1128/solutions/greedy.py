# TIER: greedy
"""Obvious recipe: "start immediately, keep feeding." For every tank that
gets used, the only start day ever considered is day 0 -- no phase/timing
reasoning at all. Under a flour budget, repeatedly activate whichever
still-idle tank gives the best newly-covered order value per unit flour if
fed continuously from day 0 through the end of the horizon, until the
budget runs out. This recipe pays for LEVEL (feed hard, feed early) and is
blind to LANDING SPOT -- it cannot notice that a tank's own pulse cycle
might be crashed-and-low on exactly the days that matter."""
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


def value_covered(cont_i, orders, covered):
    """Orders satisfied if fed continuously from day 0 (s fixed at 0)."""
    total = 0
    newly = []
    for j, (day, theta, val) in enumerate(orders):
        if covered[j]:
            continue
        if cont_i[day] >= theta:
            total += val
            newly.append(j)
    return total, newly


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

    covered = [False] * M
    used = [False] * T
    remaining = BUDGET
    schedule = []

    while True:
        best = None  # (ratio, tank, cost, val, newly)
        for i in sorted(range(T), key=lambda x: (costs[x], x)):
            if used[i]:
                continue
            cost = H * costs[i]
            if cost <= 0 or cost > remaining:
                continue
            val, newly = value_covered(conts[i], orders, covered)
            if val <= 0:
                continue
            ratio = val / cost
            if best is None or ratio > best[0] + 1e-12:
                best = (ratio, i, cost, val, newly)
        if best is None:
            break
        _, i, cost, val, newly = best
        used[i] = True
        remaining -= cost
        for j in newly:
            covered[j] = True
        schedule.extend((i, d) for d in range(H))

    lines = [str(len(schedule))]
    for (i, d) in schedule:
        lines.append(f"{i} {d}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
