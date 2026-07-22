#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE Fair-Share Capacity Market instance to stdout.
Deterministic: fully seeded by testId (fixed handcrafted topologies for the hub-style
cases, and a seeded PRNG for the small generic cases). No wall-time / no external state.
"""
import sys, random


def emit(E, flows, cap, cost, budget):
    out = []
    out.append(f"{E} {len(flows)} {budget:.6f}")
    for e in range(E):
        out.append(f"{cap[e]:.6f} {cost[e]:.6f}")
    for route in flows:
        out.append(f"{len(route)} " + " ".join(str(x) for x in route))
    sys.stdout.write("\n".join(out) + "\n")


def hub(K, hub_cap, priv_caps, budget, hub_cost=1.0, priv_costs=None, hub_id=0, priv_start=1):
    """One shared hub link + K private links, each private link crossed by exactly
    one flow together with the hub -- the classic shared-link-coupling topology."""
    if priv_costs is None:
        priv_costs = [1.0] * K
    E = priv_start + K
    flows = [[hub_id, priv_start + i] for i in range(K)]
    cap = {hub_id: hub_cap}
    cost = {hub_id: hub_cost}
    for i in range(K):
        cap[priv_start + i] = priv_caps[i]
        cost[priv_start + i] = priv_costs[i]
    return E, flows, cap, cost, budget


def two_hub(K1, hub1_cap, priv1_caps, K2, hub2_cap, priv2_caps, budget,
            hub_cost=1.0, priv_cost=1.0):
    """Two INDEPENDENT hubs (disjoint flow groups) -- tests whether the solver can
    correctly rank scarcity ACROSS separate coupled sub-networks, not just within one."""
    E = 2 + K1 + K2
    flows = []
    cap = {0: hub1_cap, 1: hub2_cap}
    cost = {0: hub_cost, 1: hub_cost}
    idx = 2
    for i in range(K1):
        flows.append([0, idx]); cap[idx] = priv1_caps[i]; cost[idx] = priv_cost; idx += 1
    for i in range(K2):
        flows.append([1, idx]); cap[idx] = priv2_caps[i]; cost[idx] = priv_cost; idx += 1
    return E, flows, cap, cost, budget


def gen_generic(seed, E, F, maxlen=2, cap_lo=3, cap_hi=15, cost_lo=1, cost_hi=5,
                 budget_frac=0.35):
    rnd = random.Random(seed)
    flows = []
    for _ in range(F):
        L = rnd.randint(1, maxlen)
        flows.append(sorted(rnd.sample(range(E), min(L, E))))
    cap = {e: rnd.uniform(cap_lo, cap_hi) for e in range(E)}
    cost = {e: rnd.uniform(cost_lo, cost_hi) for e in range(E)}
    budget = budget_frac * sum(cap.values())
    return E, flows, cap, cost, budget


def case(tid):
    if tid == 1:
        return gen_generic(101, E=4, F=3, maxlen=2, budget_frac=0.4)
    if tid == 2:
        return gen_generic(202, E=5, F=5, maxlen=2, budget_frac=0.35)
    if tid == 3:  # a genuine single bottleneck: the hub really is scarce
        return hub(4, hub_cap=4.0, priv_caps=[10, 9, 11, 8], budget=8.0)
    if tid == 4:  # decoy trap onset: hub looks busy but is already ample
        return hub(4, hub_cap=40.0, priv_caps=[1, 1.4, 1.8, 2.2], budget=16.0)
    if tid == 5:  # decoy trap, stronger: symmetric private bottlenecks
        return hub(6, hub_cap=70.0, priv_caps=[1.0] * 6, budget=20.0)
    if tid == 6:  # decoy trap, strongest + heterogeneous private scarcity
        return hub(6, hub_cap=90.0, priv_caps=[0.6, 0.8, 1.0, 1.3, 1.7, 2.2], budget=20.0)
    if tid == 7:  # a second genuine bottleneck, different scale
        return hub(5, hub_cap=6.0, priv_caps=[16, 15, 14, 17, 13], budget=10.0)
    if tid == 8:  # hybrid: one real bottleneck hub + one decoy hub, disjoint flows
        return two_hub(K1=4, hub1_cap=5.0, priv1_caps=[14, 13, 15, 12],
                        K2=5, hub2_cap=80.0, priv2_caps=[1, 1.2, 1.5, 1.8, 2.1],
                        budget=22.0)
    if tid == 9:  # large decoy trap: scale/perf test
        priv = [0.7 + 0.3 * i for i in range(10)]
        return hub(10, hub_cap=110.0, priv_caps=priv, budget=34.0)
    if tid == 10:  # largest / adversarial: hybrid at scale
        priv1 = [20 + i for i in range(8)]
        priv2 = [1.0 + 0.4 * i for i in range(10)]
        return two_hub(K1=8, hub1_cap=7.0, priv1_caps=priv1,
                        K2=10, hub2_cap=60.0, priv2_caps=priv2,
                        budget=35.0)
    raise ValueError(f"no such testId: {tid}")


def main():
    tid = int(sys.argv[1])
    E, flows, cap, cost, budget = case(tid)
    emit(E, flows, cap, cost, budget)


if __name__ == "__main__":
    main()
