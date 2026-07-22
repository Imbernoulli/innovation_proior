#!/usr/bin/env python3
"""gen.py <testId> -- night-custodian row-sweep instance generator.
Deterministic: all randomness seeded purely from testId.
"""
import sys, random
from collections import Counter


def compute_req(T, F0, Fmin, demand_times):
    """req[t] = minimum cumulative number of sweeps that must have completed by
    round t (inclusive) to keep the free-row stock >= Fmin at every round, assuming
    sweeps happen exactly when first forced (reactive, one per shortfall)."""
    dc = Counter(demand_times)
    req = [0] * (T + 1)
    free = F0
    cum = 0
    for t in range(1, T + 1):
        free -= dc.get(t, 0)
        while free < Fmin:
            free += 1
            cum += 1
        req[t] = cum
    return req


def forced_rounds_from_req(req, T):
    return [t for t in range(1, T + 1) if req[t] > req[t - 1]]


def build_instance(test_id):
    N_list = [6, 7, 7, 8, 8, 9, 9, 10, 10, 11]
    T_list = [12, 13, 14, 15, 16, 17, 18, 19, 20, 22]
    KFRAC = [0.50, 0.50, 0.55, 0.60, 0.60, 0.65, 0.65, 0.70, 0.70, 0.75]
    TRAP_TESTS = {2, 3, 4, 5, 6, 7, 8, 9, 10}

    idx = test_id - 1
    N = N_list[idx]
    T = T_list[idx]
    F0, Fmin = 3, 1
    slack = F0 - Fmin

    rng = random.Random(1000003 * test_id + 7919)

    K_target = max(3, min(N - 1, round(KFRAC[idx] * N)))
    D = min(K_target + slack, T - 1)
    K_target = max(1, D - slack)

    # spread D distinct demand rounds across [2, T] with light jitter
    base_positions = (
        [round(2 + i * (T - 2) / max(1, D - 1)) for i in range(D)] if D > 1 else [max(2, T // 2)]
    )
    demand_set = set()
    for p in base_positions:
        p = min(T, max(2, p + rng.choice([-1, 0, 0, 1])))
        while p in demand_set:
            p += 1
            if p > T:
                p = 2
                while p in demand_set:
                    p += 1
        demand_set.add(p)
    demand_times = sorted(demand_set)

    req = compute_req(T, F0, Fmin, demand_times)
    K_needed = req[T]
    fr = forced_rounds_from_req(req, T)
    # if the box office overbooked capacity (K_needed > N rows exist), trim demand
    while K_needed > N and len(demand_times) > 1:
        demand_times.pop()
        req = compute_req(T, F0, Fmin, demand_times)
        K_needed = req[T]
        fr = forced_rounds_from_req(req, T)

    segs = [None] * N

    is_trap_test = test_id in TRAP_TESTS
    n_cheap = 2 if N <= 7 else 3

    n_traps = 0
    if is_trap_test and K_needed - n_cheap >= 2:
        target = max(1, K_needed - n_cheap)
        # need >= n_traps expensive-flat fillers left over to absorb the vacated
        # early slots, i.e. N - n_cheap - n_traps >= n_traps
        cap_by_fillers = N - n_cheap - 1
        cap_by_deadline = K_needed - n_cheap - 1
        n_traps = max(0, min(target, cap_by_fillers, cap_by_deadline))

    order = list(range(N))
    rng.shuffle(order)
    trap_idx = order[:n_traps]
    cheap_idx = order[n_traps:n_traps + n_cheap]
    expensive_idx = order[n_traps + n_cheap:]

    TRAP_K = 10
    CHEAP_K_CHOICES = [1, 2]
    EXPENSIVE_K_CHOICES = [12, 13, 14]

    # trap rows: attractively cheap NOW, then almost all patrons leave together later.
    # A myopic sweeper first considers trap m at forced round fr[n_cheap+m-1] (right after
    # the always-cheap rows run out) -- if swept then it costs TRAP_K. An optimal planner
    # instead sweeps an always-expensive row at that slot and saves trap m for one of the
    # LAST n_traps forced rounds, by which point its cluster has already resolved (cost 0).
    # So the death cluster must land no later than that late "tail" slot, and strictly
    # after the early "temptation" slot -- otherwise there is no legal window to dodge into.
    for m, i in enumerate(trap_idx, start=1):
        early_rank = min(n_cheap + m, len(fr))
        r_anchor = fr[early_rank - 1] if fr else 2
        tail_rank = min(max(K_needed - n_traps + m, early_rank + 1), len(fr))
        tail_round = fr[tail_rank - 1] if fr else T
        c = tail_round if tail_round > r_anchor else min(T, r_anchor + 1)
        segs[i] = [c] * TRAP_K

    for i in cheap_idx:
        k = rng.choice(CHEAP_K_CHOICES)
        segs[i] = [-1] * k  # always-occupied but very few seats -> cheap to sweep any time

    for i in expensive_idx:
        k = rng.choice(EXPENSIVE_K_CHOICES)
        segs[i] = [-1] * k  # a packed, stable row -> costly to sweep whenever touched

    return N, T, F0, Fmin, segs, demand_times


def main():
    test_id = int(sys.argv[1])
    N, T, F0, Fmin, segs, demand_times = build_instance(test_id)
    out = []
    out.append(f"{N} {T} {F0} {Fmin}")
    for i in range(N):
        d = segs[i]
        out.append(str(len(d)) + " " + " ".join(str(x) for x in d))
    out.append(str(len(demand_times)))
    out.append(" ".join(str(x) for x in demand_times))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
