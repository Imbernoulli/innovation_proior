# TIER: trivial
"""Never touch the shared backbone at all: solve every commodity's min-cost
flow using ONLY its non-trunk edges. Always feasible (global usage = 0 <= C)
but ignores the cheap trunk routes entirely -- this reproduces the checker's
own baseline B exactly."""
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


def main():
    commodities, _C = read_instance()
    out = []
    for c in commodities:
        edges_notrunk = [(u, v, (cap if shared == 0 else 0), cost)
                          for (u, v, cap, cost, shared, weight) in c["edges"]]
        _achieved, flows = mcmf(c["n"], edges_notrunk, c["s"], c["t"], c["d"])
        out.extend(str(f) for f in flows)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
