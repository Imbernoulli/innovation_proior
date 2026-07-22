#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for Fair-Share Capacity Market.

Validates the participant's capacity-purchase artifact, then computes the proportional-fair
(log-utility) rate equilibrium TWICE via a coordinate-descent (block Gauss-Seidel with exact
per-coordinate bisection) dual solver -- once for the participant's purchase, once for the
checker's own "spend proportional to existing capacity" reference -- and prints the ratio.
No randomness anywhere; fixed sweep/iteration counts -> bit-for-bit deterministic on reruns.
"""
import sys, math


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
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
    return E, F, budget, cap, cost, flows


def solve_equilibrium(flows, C, sweeps=60, bisect_iters=55):
    """Block coordinate descent on the NUM dual: for each link e (holding all other link
    prices fixed), the 1-D optimality condition sum_{f crossing e} 1/(p_e + rest_f) = C_e is
    monotonically DECREASING in p_e >= 0 -> solved exactly by bisection each sweep. This is
    the classic iterative-water-filling / congestion-pricing update for proportional
    fairness; convex + coordinatewise-exact => converges reliably for these small networks.
    """
    active = sorted(set(e for route in flows for e in route))
    if not active:
        return 0.0
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
            return float('-inf')
        val += math.log(1.0 / s)
    return val


def value_of(flows, cap, x, E):
    C = {e: cap[e] + x.get(e, 0.0) for e in range(E)}
    return solve_equilibrium(flows, C)


def flowcount_of(flows, E):
    fc = {e: 0 for e in range(E)}
    for route in flows:
        for e in route:
            fc[e] += 1
    return fc


def capacity_proportional_alloc(flows, cap, cost, budget, E):
    fc = flowcount_of(flows, E)
    active = [e for e in range(E) if fc[e] > 0]
    totcap = sum(cap[e] for e in active)
    x = {}
    for e in active:
        dollars = budget * (cap[e] / totcap)
        x[e] = dollars / cost[e]
    return x


def fail(msg):
    print(f"INFEASIBLE: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]
    E, F, budget, cap, cost, flows = read_instance(in_path)

    try:
        with open(out_path) as f:
            out_toks = f.read().split()
    except Exception:
        fail("cannot read output")

    if len(out_toks) != E:
        fail(f"expected {E} tokens, got {len(out_toks)}")

    x = {}
    total_cost = 0.0
    for e in range(E):
        try:
            v = float(out_toks[e])
        except ValueError:
            fail(f"token {e} not a float")
        if not math.isfinite(v):
            fail(f"token {e} not finite")
        if v < -1e-9:
            fail(f"negative purchase x_{e}={v}")
        v = max(0.0, v)
        x[e] = v
        total_cost += v * cost[e]

    if total_cost > budget * (1.0 + 1e-6) + 1e-6:
        fail(f"over budget: spent {total_cost:.6f} > budget {budget:.6f}")

    V0 = value_of(flows, cap, {}, E)
    if not math.isfinite(V0):
        fail("degenerate instance (V0 non-finite)")

    F_you = value_of(flows, cap, x, E)
    if not math.isfinite(F_you):
        fail("purchase leaves a flow with zero equilibrium rate")
    F_gain = max(0.0, F_you - V0)

    x_ref = capacity_proportional_alloc(flows, cap, cost, budget, E)
    F_ref_raw = value_of(flows, cap, x_ref, E)
    F_ref = max(0.0, F_ref_raw - V0)

    sc = min(1000.0, 100.0 * F_gain / max(1e-9, F_ref))
    print(f"V0={V0:.6f} F_you={F_you:.6f} F_gain={F_gain:.6f} F_ref={F_ref:.6f}")
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
