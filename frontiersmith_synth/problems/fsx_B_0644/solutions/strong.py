# TIER: strong
"""Insight: decompose the network by 2-edge-connectivity (Tarjan bridges).
Every BRIDGE edge is a min-cut of size 1 all by itself -- its own survival
probability is a direct multiplicative FACTOR of the whole network's
reliability, so it has no "backup" and every unit spent there is never
diluted by redundancy.  Every non-bridge edge sits inside a 2-edge-connected
"blob" (a bundle of parallel routes forming a larger min-cut); the blob only
fails if ALL its edges fail simultaneously, so once a couple of its edges are
even moderately reliable, pouring MORE budget into that same blob buys almost
nothing -- the blob's own product-of-failures is already tiny.

Because bridges and blobs act as INDEPENDENT multiplicative factors of the
overall reliability, we can evaluate the true marginal gain of any single
upgrade in O(1) (bridges) / O(blob size) (blobs) without touching the rest of
the graph, and do exact steepest-ascent hill climbing on the REAL objective
(never an isolated per-edge proxy) until the budget is spent.  This is what
correctly discovers that value concentrates on edges shared by many
near-minimum cuts (the bridges), not on whichever single path looks best in
isolation.
"""
import sys


def find_bridges(n, adj, m):
    disc = [-1] * (n + 1)
    low = [-1] * (n + 1)
    timer = [0]
    bridges = set()

    def dfs(root):
        stack = [(root, -1, iter(adj[root]))]
        disc[root] = low[root] = timer[0]; timer[0] += 1
        while stack:
            u, in_edge, it = stack[-1]
            advanced = False
            for v, eidx in it:
                if eidx == in_edge:
                    continue
                if disc[v] == -1:
                    disc[v] = low[v] = timer[0]; timer[0] += 1
                    stack.append((v, eidx, iter(adj[v])))
                    advanced = True
                    break
                else:
                    low[u] = min(low[u], disc[v])
            if not advanced:
                stack.pop()
                if stack:
                    pu, _, _ = stack[-1]
                    low[pu] = min(low[pu], low[u])
                    if low[u] > disc[pu]:
                        bridges.add(in_edge)
    for s in range(1, n + 1):
        if disc[s] == -1:
            dfs(s)
    return bridges


class DSU:
    def __init__(self, n):
        self.p = list(range(n + 1))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main():
    toks = sys.stdin.read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    m = int(toks[idx]); idx += 1
    k = int(toks[idx]); idx += 1
    B = int(toks[idx]); idx += 1
    terminals = [int(toks[idx + i]) for i in range(k)]; idx += k
    edges = []
    for _ in range(m):
        u, v, p0, c, mu = (int(toks[idx + j]) for j in range(5))
        idx += 5
        edges.append((u, v, p0, c, mu))

    adj = [[] for _ in range(n + 1)]
    for i, (u, v, p0, c, mu) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    bridges = find_bridges(n, adj, m)

    dsu = DSU(n)
    for i, (u, v, p0, c, mu) in enumerate(edges):
        if i not in bridges:
            dsu.union(u, v)

    blob_edges = {}
    for i, (u, v, p0, c, mu) in enumerate(edges):
        if i not in bridges:
            r = dsu.find(u)
            blob_edges.setdefault(r, []).append(i)

    p0f = [edges[i][2] / 1000.0 for i in range(m)]
    cost = [edges[i][3] for i in range(m)]
    maxu = [edges[i][4] for i in range(m)]

    levels = [0] * m

    def bridge_factor(i):
        return 1.0 - p0f[i] * (0.5 ** levels[i])

    def blob_factor(idxs):
        prod_dead = 1.0
        for i in idxs:
            prod_dead *= p0f[i] * (0.5 ** levels[i])
        return 1.0 - prod_dead

    def total_reliability():
        tot = 1.0
        for i in range(m):
            if i in bridges:
                tot *= bridge_factor(i)
        for idxs in blob_edges.values():
            tot *= blob_factor(idxs)
        return tot

    budget = B
    while True:
        cur = total_reliability()
        best = None
        for i in range(m):
            if levels[i] >= maxu[i] or cost[i] > budget:
                continue
            levels[i] += 1
            new = total_reliability()
            levels[i] -= 1
            delta = new - cur
            if delta <= 1e-15:
                continue
            dpc = delta / cost[i]
            if best is None or dpc > best[0]:
                best = (dpc, i)
        if best is None:
            break
        _, i = best
        levels[i] += 1
        budget -= cost[i]

    print("\n".join(str(x) for x in levels))


if __name__ == "__main__":
    main()
