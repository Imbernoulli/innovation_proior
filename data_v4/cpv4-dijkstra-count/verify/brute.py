import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx+=1
    m = int(data[idx]); idx+=1
    s = int(data[idx]); idx+=1
    MOD = 1000000007
    edges = []
    for _ in range(m):
        u = int(data[idx]); idx+=1
        v = int(data[idx]); idx+=1
        w = int(data[idx]); idx+=1
        edges.append((u, v, w))

    INF = float('inf')
    # Independent shortest distances via Bellman-Ford (weights are non-negative here,
    # but Bellman-Ford is method-independent from Dijkstra).
    dist = [INF]*n
    dist[s] = 0
    for _ in range(n):
        changed = False
        for (u, v, w) in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            break

    # Count shortest routes by DAG-DP over tight edges (dist[u]+w == dist[v]).
    # Each tight edge (including each parallel copy) is a distinct way to extend.
    # Process nodes in increasing dist order; ties broken arbitrarily but the
    # shortest-path DAG is acyclic on strictly-increasing dist, and equal-dist
    # tight edges cannot exist with positive weights, so order by dist is a valid topo order.
    # Build incoming tight edges per node.
    incoming = [[] for _ in range(n)]
    for (u, v, w) in edges:
        if dist[u] != INF and dist[u] + w == dist[v]:
            incoming[v].append(u)

    order = sorted([i for i in range(n) if dist[i] != INF], key=lambda i: dist[i])
    cnt = [0]*n
    cnt[s] = 1
    for v in order:
        if v == s:
            # source keeps its base count of 1; tight self-incoming (zero-weight
            # cycles) impossible with positive weights, so nothing to add.
            continue
        total = 0
        for u in incoming[v]:
            total += cnt[u]
        cnt[v] = total % MOD

    out = []
    for i in range(n):
        if dist[i] == INF:
            out.append("-1")
        else:
            out.append(str(cnt[i] % MOD))
    print("\n".join(out))

main()
