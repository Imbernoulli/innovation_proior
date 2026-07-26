# TIER: strong
# INSIGHT: full transition coverage is a Chinese-postman EDGE TOUR, not
# per-transition path spamming and not a myopic nearest-uncovered-edge walk.
# Every state currently has out-degree exactly k (the DFA is a total
# function); its in-degree may differ, so some states are net "surplus"
# (more transitions arrive than leave -> need extra outgoing capacity) and
# others are net "deficient" (need extra incoming capacity). We solve the
# MIN-COST assignment (via shortest-path distances) matching surplus states
# to deficient states -- also searching over which state should be the walk's
# open ENDPOINT (an open Eulerian trail beats always closing back to s0) --
# duplicate exactly those shortest paths to rebalance the transition graph,
# then read off a single Eulerian trail (Hierholzer's algorithm) that covers
# every transition exactly the minimum number of times required.
import sys
from collections import deque


def main():
    data = sys.stdin.read().split("\n")
    head = data[0].split()
    n, k, s0 = int(head[0]), int(head[1]), int(head[2])
    symbols = data[1].split()
    trans = []
    for i in range(n):
        trans.append([int(x) for x in data[2 + i].split()])

    def bfs_from(src):
        dist = [-1] * n
        parent = [None] * n
        dist[src] = 0
        dq = deque([src])
        while dq:
            u = dq.popleft()
            for s in range(k):
                v = trans[u][s]
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = (u, s)
                    dq.append(v)
        return dist, parent

    all_dist = []
    all_parent = []
    for v in range(n):
        d, p = bfs_from(v)
        all_dist.append(d)
        all_parent.append(p)

    def path_edges(a, b):
        # (state, symbol) edges of the shortest path a -> b
        if a == b:
            return []
        parent = all_parent[a]
        seq = []
        cur = b
        while cur != a:
            u, s = parent[cur]
            seq.append((u, s))
            cur = u
        seq.reverse()
        return seq

    indeg = [0] * n
    for u in range(n):
        for s in range(k):
            indeg[trans[u][s]] += 1
    orig_imb = [indeg[v] - k for v in range(n)]

    def min_cost_assignment(surplus, deficient):
        m = len(surplus)
        if m == 0:
            return 0, []
        INF = float("inf")
        full = 1 << m
        dp = [INF] * full
        dp[0] = 0
        choice = [-1] * full
        for mask in range(full):
            if dp[mask] == INF:
                continue
            i = bin(mask).count("1")
            if i >= m:
                continue
            su = surplus[i]
            row = all_dist[su]
            for j in range(m):
                if mask & (1 << j):
                    continue
                c = row[deficient[j]]
                nm = mask | (1 << j)
                nv = dp[mask] + c
                if nv < dp[nm]:
                    dp[nm] = nv
                    choice[nm] = j
        pairs = []
        mask = full - 1
        i = m
        while mask:
            j = choice[mask]
            i -= 1
            pairs.append((surplus[i], deficient[j]))
            mask ^= (1 << j)
        return dp[full - 1], pairs

    # search which state t should be the trail's open endpoint (t == s0 means
    # a closed circuit; any other t needs s0 to carry one extra "surplus" unit
    # and t one extra "deficient" unit relative to the plain balance target).
    best = None
    for t in range(n):
        x = list(orig_imb)
        if t != s0:
            x[s0] += 1
            x[t] -= 1
        surplus, deficient = [], []
        for v in range(n):
            if x[v] > 0:
                surplus += [v] * x[v]
            elif x[v] < 0:
                deficient += [v] * (-x[v])
        cost, pairs = min_cost_assignment(surplus, deficient)
        total = n * k + cost
        if best is None or total < best[0]:
            best = (total, t, pairs)

    _, t_end, pairs = best

    adjlist = [[] for _ in range(n)]
    for u in range(n):
        for s in range(k):
            adjlist[u].append((symbols[s], trans[u][s]))
    for (a, b) in pairs:
        for (u, s) in path_edges(a, b):
            adjlist[u].append((symbols[s], trans[u][s]))

    # iterative Hierholzer: consume every edge exactly once
    adjptr = [0] * n
    stack = [(s0, None)]
    circuit = []
    while stack:
        v, sym_in = stack[-1]
        if adjptr[v] < len(adjlist[v]):
            sym, u = adjlist[v][adjptr[v]]
            adjptr[v] += 1
            stack.append((u, sym))
        else:
            stack.pop()
            if sym_in is not None:
                circuit.append(sym_in)
    circuit.reverse()

    sys.stdout.write("1\n" + "".join(circuit) + "\n")


if __name__ == "__main__":
    main()
