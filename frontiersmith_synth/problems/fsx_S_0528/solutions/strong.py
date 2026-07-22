# TIER: strong
# INSIGHT (the genuine leverage, not hotspot relief):
#  (1) The trip schedule is public, so the cascade FRONTIER is predictable: for the line
#      that is about to trip, its live ring neighbours are exactly the lines the wave hits.
#  (2) The transition is unforgiving -- a tripped line dumps its WHOLE overloaded load onto
#      a single live neighbour, so letting the cascade take even one secondary hop makes it
#      run away around the ring.  The place to break it is the first ring.
#  (3) BUT the per-step shed budget is too small to open enough margin on a frontier line in
#      the single step the wave arrives.  So the margin must be opened PROACTIVELY -- shedding
#      the frontier lines across the steps LEADING UP TO the trip, latest-first so backlog
#      carryover is paid for as few steps as possible while still fitting the budget.
# A myopic hotspot-relief policy sheds the currently-most-stressed line (often an unrelated
# high-utilisation decoy), never opens the frontier margin in time, and eats the cascade.
import sys, json

inst = json.load(sys.stdin)
L, T = inst["L"], inst["T"]
cap = inst["cap"]
nbr = inst["nbr"]
schedule = inst["schedule"]
budget = float(inst["budget"])
BUF = 1.0


def rollout_loads(shed_plan):
    """Deterministic replay (mirrors the evaluator) recording the load of every line at
    the START of each step, so we can read off what each frontier line will hold when its
    wave arrives.  Returns list of per-step load snapshots and the alive history."""
    load = [float(x) for x in inst["load0"]]
    alive = [True] * L
    snaps, alives = [], []
    for t in range(T):
        snaps.append(list(load))
        alives.append(list(alive))
        row = shed_plan[t]
        raw = sum(row[l] for l in range(L) if alive[l] and row[l] > 0.0)
        f = 1.0 if raw <= budget else budget / raw
        for l in range(L):
            if alive[l]:
                s = min(row[l] * f, load[l])
                if s > 0:
                    load[l] -= s
        queue = []
        e = schedule[t]
        if 0 <= e < L and alive[e]:
            queue.append(e)
        qi = 0
        while qi < len(queue):
            x = queue[qi]; qi += 1
            if not alive[x]:
                continue
            alive[x] = False
            lx = load[x]; load[x] = 0.0
            live = [y for y in nbr[x] if alive[y]]
            if live:
                share = lx / len(live)
                for y in live:
                    load[y] += share
                    if load[y] > cap[y]:
                        queue.append(y)
    return snaps, alives


shed_plan = [[0.0] * L for _ in range(T)]
used = [0.0] * T

# One do-nothing rollout gives the frontier line loads at each trip time.
snaps, alives = rollout_loads([[0.0] * L for _ in range(T)])

for te in range(T):
    e = schedule[te]
    if not (0 <= e < L):
        continue
    load_te = snaps[te]
    alive_te = alives[te]
    if not alive_te[e]:
        continue
    live = [y for y in nbr[e] if alive_te[y]]
    if not live:
        continue
    share = load_te[e] / len(live)
    for y in live:
        need = load_te[y] + share - cap[y] + BUF     # open just enough margin on the frontier line
        if need <= 0:
            continue
        need = min(need, load_te[y])
        # allocate the shed backward from the trip step, latest-first, within the budget
        rem = need
        for t in range(te, -1, -1):
            avail = budget - used[t]
            if avail <= 0:
                continue
            take = min(rem, avail)
            shed_plan[t][y] += take
            used[t] += take
            rem -= take
            if rem <= 1e-9:
                break

print(json.dumps({"shed": shed_plan}))
