# TIER: strong
"""
Insight: a cubic metre of water is worth more the higher upstream / the
higher-head it is released from, because power = release * gain(level) and
gain is concave-increasing in level. So the goal is NOT to move water
through the chain as fast as possible (that is what greedy does, and it
collapses head to the floor gain a_i almost everywhere) -- it is to hold
each reservoir near a high-head target band for as long as possible, and
release only what has to be released.

The only reason to ever release more than that is the OTHER mechanism:
spill-avoidance. A unit that gets force-spilled earns zero energy at this
dam. Because the whole inflow series is given up front (this is an offline
planning problem), each reservoir's future arrivals are knowable exactly
once every UPSTREAM reservoir's plan is fixed -- so we plan reservoirs in
topological (upstream -> downstream) order. For reservoir i, with its full
inflow sequence now known, a backward suffix-sum tells us, at every step t,
how much of the *remaining* horizon's inflow cannot possibly be drained
later even at full turbine capacity Rmax_i; exactly that surplus is
subtracted from the target level as a proactive buffer -- i.e. we start
lowering the target *before* a flood arrives, spread over many steps
(each capped at Rmax_i) instead of after it arrives (when a single step
cannot possibly catch up). This is the "potential-energy ledger" argument:
price water against its own future capacity needs, not against this
instant's power alone. Greedy has no way to discover this because it never
looks past the current step, let alone downstream capacity.
"""
import sys


def plan_reservoir(C_i, Rmax_i, delay_i, init_i, T, inflow_seq):
    lo_bound = 0.45 * C_i
    hi_bound = 0.85 * C_i
    trickle_frac = 0.05  # keep some flow moving even below target

    suffix = [0.0] * (T + 2)
    for t in range(T - 1, -1, -1):
        suffix[t] = suffix[t + 1] + inflow_seq[t]

    level = init_i
    release = [0.0] * T
    spill = [0.0] * T
    for t in range(T):
        remaining_steps = T - 1 - t  # steps strictly after t
        future_inflow = suffix[t + 1]
        deficit = future_inflow - Rmax_i * remaining_steps
        target = C_i - max(0.0, deficit)
        if target < lo_bound:
            target = lo_bound
        elif target > hi_bound:
            target = hi_bound
        # end-game feasibility: never plan to still be holding more than
        # what the turbine could possibly drain in the steps left, or the
        # leftover water strands at the end with zero energy credit.
        drainable = Rmax_i * remaining_steps
        if target > drainable:
            target = drainable

        inflow_it = inflow_seq[t]
        pre_if_no_release = level + inflow_it
        cap = min(Rmax_i, level)
        shed = pre_if_no_release - target
        if shed > 1e-9:
            # above the head-preserving target (or a flood is being pre-
            # buffered for): release exactly the excess, rate-limited.
            rel = min(shed, cap)
        else:
            # below target: still keep a modest trickle flowing instead of
            # hoarding the entire horizon away, so short/low-inflow cases
            # keep generating energy while the level climbs toward target.
            rel = min(trickle_frac * level, cap)

        pre = level - rel + inflow_it
        if pre > C_i:
            sp = pre - C_i
            new_level = C_i
        else:
            sp = 0.0
            new_level = max(0.0, pre)

        release[t] = rel
        spill[t] = sp
        level = new_level
    return release, spill


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); T = int(next(it))
    C = []; Rmax = []; DELAY = []; INIT = []
    for _ in range(N):
        c = float(next(it)); rmax = float(next(it))
        next(it); next(it)  # a_i, b_i not needed by the level-band heuristic
        d = int(next(it))
        init = float(next(it))
        C.append(c); Rmax.append(rmax); DELAY.append(d); INIT.append(init)
    local_inflow = []
    for _ in range(N):
        local_inflow.append([float(next(it)) for _ in range(T)])

    passthrough = [[0.0] * T for _ in range(N)]
    release_all = [None] * N

    for i in range(N):
        eff_inflow = list(local_inflow[i])
        if i > 0:
            d = DELAY[i]
            for t in range(T):
                s = t - d
                if s >= 0:
                    eff_inflow[t] += passthrough[i - 1][s]
        rel, sp = plan_reservoir(C[i], Rmax[i], DELAY[i], INIT[i], T, eff_inflow)
        release_all[i] = rel
        for t in range(T):
            passthrough[i][t] = rel[t] + sp[t]

    out = []
    for i in range(N):
        out.append(" ".join("%.6f" % v for v in release_all[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
