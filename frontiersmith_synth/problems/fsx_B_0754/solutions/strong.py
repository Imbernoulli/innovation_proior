# TIER: strong
"""The insight: under proportional fairness, a link's TRUE marginal value is not its raw
flow count but the sum of shadow prices of the flows crossing it -- a quantity that only
emerges from solving the WHOLE network's rate equilibrium. This solution re-solves that
equilibrium (same coordinate-descent dual solver the checker uses) after every increment
of spending, and always buys the next slice of budget wherever the CURRENT shadow price
per dollar is highest -- a discretized greedy ascent on the (concave) utility-gain
function of the purchase, which is provably well-behaved because V(capacity) is concave.
This is fundamentally different from greedy.py: it reacts to the CURRENT state of the
network instead of committing everything up front to a single, statically-ranked link."""
import sys, math


def solve_equilibrium(flows, C, sweeps=55, bisect_iters=50):
    active = sorted(set(e for route in flows for e in route))
    if not active:
        return 0.0, {}
    flows_of_link = {e: [] for e in active}
    for fi, route in enumerate(flows):
        for e in route:
            flows_of_link[e].append(fi)
    p = {e: 0.5 for e in active}
    for _ in range(sweeps):
        for e in active:
            rests = []
            for fi in flows_of_link[e]:
                route = flows[fi]
                s = 0.0
                for e2 in route:
                    if e2 != e:
                        s += p[e2]
                rests.append(s)
            Ce = C[e]

            def h(x):
                tot = 0.0
                for r in rests:
                    d = x + r
                    tot += (1.0 / d) if d > 1e-12 else 1e18
                return tot - Ce

            if h(0.0) <= 0:
                p[e] = 0.0
                continue
            lo, hi = 0.0, 1.0
            tries = 0
            while h(hi) > 0 and tries < 200:
                hi *= 2.0
                tries += 1
            for _ in range(bisect_iters):
                mid = 0.5 * (lo + hi)
                if h(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            p[e] = 0.5 * (lo + hi)
    val = 0.0
    for route in flows:
        s = sum(p[e] for e in route)
        if s <= 1e-15:
            return float('-inf'), p
        val += math.log(1.0 / s)
    return val, p


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    E = int(next(it)); F = int(next(it)); budget = float(next(it))
    cap = {}; cost = {}
    for e in range(E):
        cap[e] = float(next(it)); cost[e] = float(next(it))
    flows = []
    for _ in range(F):
        L = int(next(it))
        route = [int(next(it)) for _ in range(L)]
        flows.append(route)

    active = sorted(set(e for route in flows for e in route))
    x = {e: 0.0 for e in range(E)}
    remaining = budget
    rounds, frac = 60, 0.08

    for _ in range(rounds):
        if remaining <= 1e-9 or not active:
            break
        C = {e: cap[e] + x[e] for e in range(E)}
        _, p = solve_equilibrium(flows, C)
        best_e = max(active, key=lambda e: (p.get(e, 0.0) / cost[e], -e))
        if p.get(best_e, 0.0) <= 1e-12:
            break
        db = remaining * frac
        if db <= 1e-9:
            break
        x[best_e] += db / cost[best_e]
        remaining -= db

    if remaining > 1e-9 and active:
        C = {e: cap[e] + x[e] for e in range(E)}
        _, p = solve_equilibrium(flows, C)
        best_e = max(active, key=lambda e: (p.get(e, 0.0) / cost[e], -e))
        x[best_e] += remaining / cost[best_e]
        remaining = 0.0

    print(" ".join("%.9f" % x[e] for e in range(E)))


if __name__ == "__main__":
    main()
