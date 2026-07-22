#!/usr/bin/env python3
"""gen.py <testId> -- one instance of outbreak-prepositioned-stockpile (fsx_A_0783).

Deterministic: all randomness seeded ONLY from testId via random.Random(testId*997+13).

Instance = a mobility graph over N cities + K outbreak scenarios (seed city, seasonal
contact multiplier). testId in {4,6,8,9,10} plant a "clustered hub + cheap bridge"
topology: population-heavy hub clusters joined only through low-population bridge
cities. A population-proportional allocation spends its whole budget inside one hub
cluster and never touches the bridges, so cross-cluster cascades among the OTHER
scenarios stay uncontained; an allocation that recognizes the bridges as multi-scenario
graph separators contains every cluster's outbreak for a fraction of the cost.
"""
import sys, math

def emit(N, K, T, alpha_percent, budget, pops, edges, scenarios):
    out = []
    out.append(f"{N} {K} {T} {alpha_percent} {budget}")
    for p in pops:
        out.append(str(p))
    out.append(str(len(edges)))
    for (u, v, w) in edges:
        out.append(f"{u} {v} {w}")
    for (s, bpct) in scenarios:
        out.append(f"{s} {bpct}")
    sys.stdout.write("\n".join(out) + "\n")

def ceil_cost(pop, alpha_percent):
    return -(-pop * alpha_percent // 100)

def build_random(rng, N, extra_edge_frac, wlo, whi, plo, phi):
    """Connected random graph: random spanning tree + extra random edges."""
    pops = [rng.randint(plo, phi) for _ in range(N)]
    order = list(range(N))
    rng.shuffle(order)
    edges = []
    edge_set = set()
    for idx in range(1, N):
        u = order[idx]
        v = order[rng.randint(0, idx - 1)]
        w = rng.randint(wlo, whi)
        a, b = (u, v) if u < v else (v, u)
        edges.append((a, b, w))
        edge_set.add((a, b))
    extra = max(0, int(N * extra_edge_frac))
    tries = 0
    while extra > 0 and tries < extra * 20:
        tries += 1
        u = rng.randint(0, N - 1)
        v = rng.randint(0, N - 1)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edge_set:
            continue
        edge_set.add((a, b))
        edges.append((a, b, rng.randint(wlo, whi)))
        extra -= 1
    return pops, edges

def build_trap(rng, clusters, hub_size, bridge_plo, bridge_phi,
                hub_plo, hub_phi, hub_wlo, hub_whi, bridge_wlo, bridge_whi):
    """`clusters` dense hub clusters chained by one cheap low-population bridge city
    between consecutive clusters. Bridges are the multi-scenario graph separators."""
    pops = []
    edges = []
    cluster_start = []
    idx = 0
    for c in range(clusters):
        cluster_start.append(idx)
        for _ in range(hub_size):
            pops.append(rng.randint(hub_plo, hub_phi))
            idx += 1
        members = list(range(cluster_start[c], cluster_start[c] + hub_size))
        for i in members:
            for j in members:
                if i < j:
                    edges.append((i, j, rng.randint(hub_wlo, hub_whi)))
    bridges = []
    for c in range(clusters - 1):
        pops.append(rng.randint(bridge_plo, bridge_phi))
        b = idx
        idx += 1
        bridges.append(b)
        m1 = rng.choice(range(cluster_start[c], cluster_start[c] + hub_size))
        m2 = rng.choice(range(cluster_start[c + 1], cluster_start[c + 1] + hub_size))
        edges.append((min(b, m1), max(b, m1), rng.randint(bridge_wlo, bridge_whi)))
        edges.append((min(b, m2), max(b, m2), rng.randint(bridge_wlo, bridge_whi)))
    N = idx
    return N, pops, edges, cluster_start, bridges, hub_size

def scenarios_random(rng, N, pops, K, blo, bhi):
    """Seeds are drawn from the upper half of cities by population (major hubs are the
    plausible outbreak origins), so directly vaccinating "the seed" is never a cheap
    shortcut -- a solver must route around via the network, not just buy out the seed."""
    ranked = sorted(range(N), key=lambda i: -pops[i])
    pool = ranked[: max(1, N // 2)]
    return [(rng.choice(pool), rng.randint(blo, bhi)) for _ in range(K)]

def scenarios_trap(rng, cluster_start, hub_size, K, blo, bhi):
    """Seed scenarios spread across ALL clusters (incl. peripheral, non-hub-center
    members) so the worst outbreak is not confined to whichever cluster a naive
    population-first allocation happens to protect."""
    clusters = len(cluster_start)
    sc = []
    for k in range(K):
        c = k % clusters
        member = cluster_start[c] + rng.randint(0, hub_size - 1)
        sc.append((member, rng.randint(blo, bhi)))
    return sc

def main():
    testId = int(sys.argv[1])
    rng = __import__("random").Random(testId * 997 + 13)

    if testId in (4, 6, 8, 9, 10):
        cfgs = {
            4:  dict(clusters=3, hub_size=6, T=25, alpha=15, bfrac=(13, 15)),
            6:  dict(clusters=3, hub_size=7, T=28, alpha=15, bfrac=(12, 14)),
            8:  dict(clusters=4, hub_size=6, T=30, alpha=14, bfrac=(11, 13)),
            9:  dict(clusters=4, hub_size=7, T=30, alpha=14, bfrac=(10, 12)),
            10: dict(clusters=5, hub_size=7, T=32, alpha=13, bfrac=(9, 11)),
        }[testId]
        N, pops, edges, cluster_start, bridges, hub_size = build_trap(
            rng, cfgs["clusters"], cfgs["hub_size"],
            bridge_plo=40, bridge_phi=160,
            hub_plo=900, hub_phi=3200,
            hub_wlo=25, hub_whi=70,
            bridge_wlo=160, bridge_whi=260)
        K = cfgs["clusters"] + (1 if testId >= 8 else 0)
        scenarios = scenarios_trap(rng, cluster_start, hub_size, K, 90, 130)
        alpha_percent = cfgs["alpha"]
        costs = [ceil_cost(p, alpha_percent) for p in pops]
        total_cost = sum(costs)
        lo, hi = cfgs["bfrac"]
        frac = rng.randint(lo, hi) / 100.0
        budget = max(1, int(total_cost * frac))
        T = cfgs["T"]
        emit(N, K, T, alpha_percent, budget, pops, edges, scenarios)
        return

    sizes = {1: 12, 2: 16, 3: 20, 5: 24, 7: 30}
    N = sizes[testId]
    K = {1: 3, 2: 3, 3: 4, 5: 4, 7: 5}[testId]
    T = {1: 18, 2: 18, 3: 20, 5: 20, 7: 22}[testId]
    pops, edges = build_random(rng, N, extra_edge_frac=1.5, wlo=20, whi=80,
                                plo=300, phi=3000)
    scenarios = scenarios_random(rng, N, pops, K, 80, 140)
    alpha_percent = rng.randint(14, 20)
    costs = [ceil_cost(p, alpha_percent) for p in pops]
    total_cost = sum(costs)
    frac = rng.randint(12, 16) / 100.0
    budget = max(1, int(total_cost * frac))
    emit(N, K, T, alpha_percent, budget, pops, edges, scenarios)

if __name__ == "__main__":
    main()
