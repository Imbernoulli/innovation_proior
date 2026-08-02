# TIER: greedy
# The obvious "smart" recipe: at every step, load-balance the zones that are
# currently departing across the reachable exits as if EVERYONE who departs
# will actually follow the directive (full compliance), always picking
# whichever exit looks least loaded relative to its capacity right now. This
# is exactly "optimal-under-full-compliance routing": textbook greedy bin
# packing, recomputed fresh every step from the evolving remaining-population
# state. It never accounts for partial compliance and never tries to keep a
# zone's directive stable, so as the remaining-population state drifts
# (zones with tight egress caps drain differently from generous ones) the
# "best" exit for a zone silently changes step to step -- contradictory
# guidance that this heuristic never notices it is giving.
import sys, json


def main():
    inst = json.load(sys.stdin)
    Z, E, T = inst["n_zones"], inst["n_exits"], inst["T"]
    cap = inst["capacity"]; reach = inst["reachable"]
    pop = inst["population"]; egress = inst["egress_cap"]

    remaining = list(pop)
    guidance = [[0] * T for _ in range(Z)]
    for t in range(T):
        depart = [min(remaining[i], egress[i]) for i in range(Z)]
        cum_load = [0.0] * E
        order = sorted(range(Z), key=lambda i: -depart[i])
        for i in order:
            if depart[i] <= 0:
                guidance[i][t] = min(e for e in range(E) if reach[i][e])
                continue
            best = min((e for e in range(E) if reach[i][e]),
                       key=lambda e: cum_load[e] / cap[e])
            guidance[i][t] = best
            cum_load[best] += depart[i]   # assumes the WHOLE departing cohort complies
        for i in range(Z):
            remaining[i] -= depart[i]

    print(json.dumps({"guidance": guidance}))


main()
