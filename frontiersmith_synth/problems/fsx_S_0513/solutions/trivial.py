# TIER: trivial
# Naive router that ignores congestion: keep ONLY the free-flow shortest path and
# push all demand down it. Reproduces the checker's single-path baseline (~0.1).
import sys, heapq


def dijkstra(N, adj, src, lat):
    INF = float('inf')
    dist = [INF] * N; pred = [-1] * N
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for (ei, v) in adj[u]:
            nd = d + lat[ei]
            if nd < dist[v] - 1e-15:
                dist[v] = nd; pred[v] = ei
                heapq.heappush(pq, (nd, v))
    return dist, pred


def main():
    raw = sys.stdin.read().split()
    it = iter(raw)
    N = int(next(it)); M = int(next(it))
    s = int(next(it)); t = int(next(it)); k = int(next(it)); D = float(next(it))
    edges = []
    for _ in range(M):
        u = int(next(it)); v = int(next(it)); a = float(next(it)); b = float(next(it))
        edges.append((u, v, a, b))
    adj = [[] for _ in range(N)]
    for ei, (u, v, a, b) in enumerate(edges):
        adj[u].append((ei, v))
    lat0 = [edges[i][2] for i in range(M)]
    dist, pred = dijkstra(N, adj, s, lat0)
    keep = []
    vv = t
    while vv != s:
        ei = pred[vv]; keep.append(ei + 1); vv = edges[ei][0]
    print(" ".join(map(str, keep)))


main()
