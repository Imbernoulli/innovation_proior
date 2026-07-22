# TIER: greedy
# The obvious recipe: relieve the CURRENT hotspot.  At every step, before the scheduled
# trip fires, find the live line with the highest utilisation (load/cap) and shed it down
# to a safe target utilisation.  It reacts to present stress and never looks at the
# (public) trip schedule, so on a fragile-corridor instance it wastes its effort on a
# high-utilisation decoy and still eats the cascade that hits the corridor elsewhere.
import sys, json

inst = json.load(sys.stdin)
L, T = inst["L"], inst["T"]
cap = inst["cap"]
nbr = inst["nbr"]
schedule = inst["schedule"]

# The candidate must plan the whole episode up front, so mirror the deterministic
# transition locally and act myopically at each simulated step.
load = [float(x) for x in inst["load0"]]
alive = [True] * L
shed_plan = []
TARGET = 0.70   # relieve the hotspot down to 70% utilisation

for t in range(T):
    row = [0.0] * L
    # pick the most-stressed live line and shed it to TARGET
    best, bu = -1, -1.0
    for l in range(L):
        if alive[l] and cap[l] > 0:
            u = load[l] / cap[l]
            if u > bu:
                bu = u; best = l
    if best >= 0 and bu > TARGET:
        s = load[best] - TARGET * cap[best]
        if s > 0:
            row[best] = s
    shed_plan.append(row)

    # replay the transition so the next step's hotspot is computed on the true state
    for l in range(L):
        if alive[l]:
            s = row[l]
            if s > load[l]:
                s = load[l]
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

print(json.dumps({"shed": shed_plan}))
