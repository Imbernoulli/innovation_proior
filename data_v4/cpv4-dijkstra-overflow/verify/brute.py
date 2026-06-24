import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    edges = []
    adj = [[] for _ in range(n + 1)]
    for _ in range(m):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        w = int(data[idx]); idx += 1
        edges.append((u, v, w))
        adj[u].append((v, w))

    # Independent method: Bellman-Ford style relaxation.
    # All weights are non-negative, no negative cycles, so distances stabilize
    # after at most n-1 full passes. We use unbounded Python ints, so there is
    # no overflow at all; this is the trustworthy oracle.
    INF = float('inf')
    dist = [INF] * (n + 1)
    dist[1] = 0
    # Bellman-Ford: relax all edges (n-1) times.
    for _ in range(n - 1 if n >= 1 else 0):
        changed = False
        for (u, v, w) in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            break

    if dist[n] == INF:
        print(-1)
    else:
        print(dist[n])

main()
