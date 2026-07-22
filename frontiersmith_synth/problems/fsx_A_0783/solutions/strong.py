# TIER: strong
"""Insight: a dosed city is a graph separator for whichever scenario's spread would
otherwise route through it. For each scenario, build the shortest-mobility-path tree
from its seed (edge length = 1/contact-weight, so "closer" == "faster spreading
route"); a city's "cut value" for that scenario is the total population hanging off it
in that tree (everyone whose only fast route to the seed passes through it). Summing
cut value across scenarios (weighted by each scenario's danger level) turns per-city
importance into a HITTING-SET score: a cheap city that sits on many scenarios'
frontiers scores far above an expensive city that only helps the one scenario seeded
near it -- exactly the reformulation a population-only heuristic cannot see. We then
knapsack-greedily buy cities by score-per-dose until the budget is exhausted, spending
any remainder on the next best partial city."""
import sys, heapq, math

def ceil_cost(pop, alpha_percent):
    return -(-pop * alpha_percent // 100)

def shortest_path_tree(N, adj, root):
    INF = float("inf")
    dist = [INF] * N
    parent = [-1] * N
    dist[root] = 0.0
    pq = [(0.0, root)]
    visited = [False] * N
    while pq:
        d, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        for (v, wlen) in adj[u]:
            nd = d + wlen
            if nd < dist[v] - 1e-15:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, parent

def subtree_populations(N, root, parent, dist, pops):
    order = sorted(range(N), key=lambda i: -dist[i] if dist[i] != float("inf") else 0.0)
    sub = [pops[i] if dist[i] != float("inf") else 0.0 for i in range(N)]
    for i in order:
        if i == root or dist[i] == float("inf"):
            continue
        pr = parent[i]
        if pr != -1:
            sub[pr] += sub[i]
    return sub

def main():
    it = iter(sys.stdin.read().split())
    def nx():
        return next(it)
    N = int(nx()); K = int(nx()); T = int(nx())
    alpha_percent = int(nx()); budget = int(nx())
    pops = [int(nx()) for _ in range(N)]
    M = int(nx())
    edges = []
    for _ in range(M):
        u = int(nx()); v = int(nx()); w = int(nx())
        edges.append((u, v, w))
    scenarios = []
    for _ in range(K):
        s = int(nx()); bpct = int(nx())
        scenarios.append((s, bpct))

    costs = [ceil_cost(pops[i], alpha_percent) for i in range(N)]

    adj_len = [[] for _ in range(N)]   # edge length = 1/weight for Dijkstra
    for (u, v, w) in edges:
        length = 1.0 / w
        adj_len[u].append((v, length))
        adj_len[v].append((u, length))

    score = [0.0] * N
    for (seed, bpct) in scenarios:
        dist, parent = shortest_path_tree(N, adj_len, seed)
        sub = subtree_populations(N, seed, parent, dist, pops)
        danger = bpct / 100.0
        for i in range(N):
            score[i] += danger * sub[i]

    ratio = []
    for i in range(N):
        c = costs[i] if costs[i] > 0 else 1
        ratio.append((score[i] / c, i))
    ratio.sort(key=lambda t: (-t[0], t[1]))

    doses = [0] * N
    remaining = budget
    partial_used = False
    for (_, i) in ratio:
        if remaining <= 0:
            break
        if costs[i] <= remaining:
            doses[i] = costs[i]
            remaining -= costs[i]
    if remaining > 0:
        for (_, i) in ratio:
            if doses[i] == 0:
                doses[i] = remaining
                remaining = 0
                break

    print(" ".join(str(d) for d in doses))

if __name__ == "__main__":
    main()
