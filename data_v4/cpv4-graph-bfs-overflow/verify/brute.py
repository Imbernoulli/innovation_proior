import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    w = [0] * (n + 1)
    for v in range(1, n + 1):
        w[v] = int(data[idx]); idx += 1
    adj = [[] for _ in range(n + 1)]
    for _ in range(m):
        a = int(data[idx]); idx += 1
        b = int(data[idx]); idx += 1
        adj[a].append(b)
        adj[b].append(a)

    # Independent brute-force shortest path: compute hop-distance from city 1
    # by iterative relaxation (Bellman-Ford-style flooding on unit edges).
    INF = float('inf')
    dist = [INF] * (n + 1)
    dist[1] = 0
    # Relax up to n times; on unit-weight graphs this converges to the
    # true hop-distance regardless of edge processing order.
    changed = True
    rounds = 0
    while changed and rounds <= n + 1:
        changed = False
        rounds += 1
        for u in range(1, n + 1):
            if dist[u] == INF:
                continue
            for v in adj[u]:
                if dist[u] + 1 < dist[v]:
                    dist[v] = dist[u] + 1
                    changed = True

    total = 0
    for v in range(1, n + 1):
        if dist[v] != INF:
            total += dist[v] * w[v]

    print(total)

main()
