import sys, random

# Difficulty ladder for the haul-switchback-pipeline family.
# Each test case: a single-lane corridor of M IDENTICAL-length blocks between the pit
# (node 0) and the crusher (node M), with M-1 capacity-1 pullouts at nodes 1..M-1. Every
# block takes exactly `t` ticks to cross (uniform), which is what makes the corridor's
# true pipeline cadence a clean multiple of `t` -- computable, not just readable.
#
# Fields tuned per testId:
#   M            number of single-lane blocks
#   K            fleet size (trucks) -- large enough that a single injection rate, not
#                fleet size, is what actually caps throughput
#   t            uniform block crossing time
#   tight_heat   True -> H_MAX is tight enough that a full descent needs a planted
#                mid-descent cooldown stop (a mandatory-stop trap for reactive dispatch)
#   idle_cool    cooling rate while waiting (small -> cooldown stops cost real ticks)
#
# LADDER: small/simple -> large/adversarial. >=3 late entries combine a large fleet with
# a tight heat cap, which is what punishes a dispatcher that serializes the two directions
# instead of computing a cadence both directions can safely share.
LADDER = [
    # M,  K, t,  tight_heat, idle_cool
    (3,  3,  4,  False, 3),
    (4,  4,  4,  False, 3),
    (5,  5,  5,  False, 3),
    (6,  6,  5,  True,  1),
    (7,  6,  4,  True,  1),
    (8,  7,  4,  True,  1),
    (9,  7,  4,  True,  1),
    (10, 8,  3,  True,  1),
    (11, 8,  3,  True,  1),
    (12, 9,  3,  True,  1),
]


def simulate_descent(t, g_list, H_MAX, idle_cool):
    """One truck's pit->crusher descent with just-in-time cooldown stops (each stop
    duration rounded UP to a multiple of t, so the whole plan stays on a clean grid).
    Returns (total_duration, list_of_wait_amounts_in_ticks)."""
    heat = 0
    t_cur = 0
    waits = []
    for g in g_list:
        if heat + g > H_MAX:
            need = heat + g - H_MAX
            w_ticks = -(-need // idle_cool)              # ceil div: min ticks to cool
            w = -(-w_ticks // t) * t                      # round UP to a multiple of t
            heat = max(0, heat - w * idle_cool)
            t_cur += w
            waits.append(w)
        t_cur += t
        heat += g
    return t_cur, waits


def build_case(testId):
    idx = min(max(testId, 1), len(LADDER)) - 1
    M, K, t, tight_heat, idle_cool = LADDER[idx]
    rng = random.Random(2000 * testId + 11)

    # baseline heat gain per block, with TWO clusters of "steep" blocks (two separate
    # mandatory-cooldown points per descent, so a reactive dispatcher re-discovers the
    # true cadence twice per wave instead of once)
    g_list = [1 + rng.randint(0, 1) for _ in range(M)]
    if M >= 4:
        cnt = max(2, M // 5)
        start1 = M // 4
        for j in range(cnt):
            g_list[(start1 + j) % M] = 6 + rng.randint(0, 3)
        start2 = (3 * M) // 4
        for j in range(cnt):
            g_list[(start2 + j) % M] = 6 + rng.randint(0, 3)

    heat_loss_up = 6  # ascending always cools by this much per block

    total_gain = sum(g_list)
    if tight_heat:
        H_MAX = max(g_list) + int(round(total_gain * 0.4))
    else:
        H_MAX = total_gain * 3 + 5

    # invariant: a full ascent must always be able to drain any reachable heat to 0,
    # so heat resets cleanly between cycles.
    while heat_loss_up * M < H_MAX + max(g_list):
        heat_loss_up += 1

    Ddown, waits = simulate_descent(t, g_list, H_MAX, idle_cool)
    P_cycle = 2 * Ddown  # one truck's own round trip (descent + equal-length return)

    T_horizon = P_cycle * max(6, (K + 3))

    return M, K, T_horizon, H_MAX, idle_cool, heat_loss_up, [t] * M, g_list


def main():
    testId = int(sys.argv[1])
    M, K, T_horizon, H_MAX, idle_cool, heat_loss_up, t_list, g_list = build_case(testId)
    print(M, K, T_horizon, H_MAX, idle_cool, heat_loss_up)
    print(" ".join(str(x) for x in t_list))
    print(" ".join(str(x) for x in g_list))


if __name__ == "__main__":
    main()
