import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    s = int(data[idx]); idx += 1
    w = [0] * (n + 1)
    for i in range(1, n + 1):
        w[i] = int(data[idx]); idx += 1
    edges = []
    adj = [[] for _ in range(n + 1)]
    for _ in range(m):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        edges.append((u, v))
        adj[u].append(v)
        adj[v].append(u)

    # Independent shortest-distance computation:
    # iterative Bellman-Ford style relaxation over undirected unit edges.
    INF = float('inf')
    dist = [INF] * (n + 1)
    dist[s] = 0
    # relax until no change (at most n-1 rounds for shortest paths)
    changed = True
    rounds = 0
    while changed and rounds <= n + 1:
        changed = False
        rounds += 1
        for (u, v) in edges:
            if dist[u] + 1 < dist[v]:
                dist[v] = dist[u] + 1
                changed = True
            if dist[v] + 1 < dist[u]:
                dist[u] = dist[v] + 1
                changed = True

    # group reachable nodes by distance, sum brightness per layer
    layer = {}
    for node in range(1, n + 1):
        if dist[node] != INF:
            d = dist[node]
            layer[d] = layer.get(d, 0) + w[node]

    # max over all non-empty layers (there is always at least layer 0 = source)
    best = None
    for d, total in layer.items():
        if best is None or total > best:
            best = total

    print(best)

if __name__ == "__main__":
    main()
