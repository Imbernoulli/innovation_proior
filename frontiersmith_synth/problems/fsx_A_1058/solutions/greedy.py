# TIER: greedy
"""The obvious first draft: solve each commodity's OWN min-cost flow completely
independently (ignore the global backbone -- every commodity looks cheapest on
its own), then, if that violates the backbone budget, walk the commodities in
the order they were GIVEN and cap each one's trunk usage just enough to stay
within whatever budget remains, stopping once the budget is used up.

This is a real single greedy pass with no re-optimization: it never asks WHICH
commodities most deserve the scarce backbone, it just processes them in input
order. When the input order is not sorted by "value per backbone unit" (which
it deliberately is not on most cases here), this locks in trunk usage for
low-value commodities early and starves high-value ones once budget runs out.
"""
import sys
from collections import deque


def mcmf(n, edges, s, t, need):
    graph = [[] for _ in range(n)]
    arc_of = [None] * len(edges)
    for i, (u, v, cap, cost) in enumerate(edges):
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])
        arc_of[i] = (u, len(graph[u]) - 1)
    flow = 0
    INF = float("inf")
    while flow < need:
        dist = [INF] * n
        dist[s] = 0.0
        inq = [False] * n
        pv = [None] * n
        dq = deque([s])
        inq[s] = True
        while dq:
            u = dq.popleft()
            inq[u] = False
            du = dist[u]
            for idx, arc in enumerate(graph[u]):
                v, cap, cost, _rev = arc
                if cap > 0 and du + cost < dist[v] - 1e-9:
                    dist[v] = du + cost
                    pv[v] = (u, idx)
                    if not inq[v]:
                        dq.append(v)
                        inq[v] = True
        if dist[t] == INF:
            break
        push = need - flow
        v = t
        while v != s:
            u, idx = pv[v]
            push = min(push, graph[u][idx][1])
            v = u
        v = t
        while v != s:
            u, idx = pv[v]
            graph[u][idx][1] -= push
            graph[v][graph[u][idx][3]][1] += push
            v = u
        flow += push
    flows = []
    for i in range(len(edges)):
        au, aidx = arc_of[i]
        flows.append(edges[i][2] - graph[au][aidx][1])
    return flow, flows


def read_instance():
    toks = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = int(toks[p]); p += 1; return v

    K = nxt()
    commodities = []
    for _ in range(K):
        n = nxt(); m = nxt(); s = nxt(); t = nxt(); d = nxt()
        edges = []
        for _ in range(m):
            u = nxt(); v = nxt(); cap = nxt(); cost = nxt(); shared = nxt(); weight = nxt()
            edges.append((u, v, cap, cost, shared, weight))
        commodities.append({"n": n, "s": s, "t": t, "d": d, "edges": edges})
    C = nxt()
    return commodities, C


def solve_full(c, trunk_cap_override=None):
    edges = []
    for (u, v, cap, cost, shared, weight) in c["edges"]:
        if shared and trunk_cap_override is not None:
            cap = min(cap, trunk_cap_override)
        edges.append((u, v, cap, cost))
    return mcmf(c["n"], edges, c["s"], c["t"], c["d"])


def usage_of(c, flows):
    u = 0
    for (edge, f) in zip(c["edges"], flows):
        if edge[4]:
            u += f * edge[5]
    return u


def main():
    commodities, C = read_instance()

    # Step 1: each commodity solved on its own, ignoring the backbone entirely.
    per_commodity_flows = []
    for c in commodities:
        _achieved, flows = solve_full(c)
        per_commodity_flows.append(flows)

    total_usage = sum(usage_of(c, f) for c, f in zip(commodities, per_commodity_flows))

    # Step 2: if that overruns the backbone, patch commodities IN INPUT ORDER.
    if total_usage > C:
        remaining = C
        for i, c in enumerate(commodities):
            u = usage_of(c, per_commodity_flows[i])
            if u <= remaining:
                remaining -= u
                continue
            weight = next((e[5] for e in c["edges"] if e[4]), 1)
            max_units = remaining // weight if weight > 0 else 0
            _achieved, flows = solve_full(c, trunk_cap_override=max_units)
            per_commodity_flows[i] = flows
            remaining -= usage_of(c, flows)
            remaining = max(0, remaining)

    out = []
    for flows in per_commodity_flows:
        out.extend(str(f) for f in flows)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
