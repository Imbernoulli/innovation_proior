# TIER: strong
# Insight: equilibrium latency is NON-MONOTONE in the edge set. Rather than trust
# every road, evaluate the actual Wardrop equilibrium and prune any edge whose
# removal strictly lowers total travel time (equilibrium-aware edge deletion).
# This finds and cuts the harmful Braess shortcuts while KEEPING the beneficial
# extra-capacity edges (which look identical topologically) -- something no purely
# structural or "keep everything" heuristic can do. It is a local search, not a
# proven optimum, so score headroom remains above it.
import sys, heapq

MAXIT = 120          # looser than the checker; comparisons are robust well before this
EPS = 1e-9


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


def equilibrium(N, edges, a, b, s, t, D, k, kept):
    idxs = list(kept)
    adj = [[] for _ in range(N)]
    for ei in idxs:
        adj[edges[ei][0]].append((ei, edges[ei][1]))

    def lat_of(ei, x):
        return a[ei] + b[ei] * (x ** k)

    lat = [0.0] * len(edges)
    for ei in idxs:
        lat[ei] = lat_of(ei, 0.0)
    dist, pred = dijkstra(N, adj, s, lat)
    if dist[t] == float('inf'):
        return None
    x = {ei: 0.0 for ei in idxs}
    v = t
    while v != s:
        ei = pred[v]; x[ei] += D; v = edges[ei][0]
    for _ in range(MAXIT):
        for ei in idxs:
            lat[ei] = lat_of(ei, x[ei])
        dist, pred = dijkstra(N, adj, s, lat)
        if dist[t] == float('inf'):
            return None
        y = {ei: 0.0 for ei in idxs}
        v = t
        while v != s:
            ei = pred[v]; y[ei] += D; v = edges[ei][0]
        cur = sum(lat[ei] * x[ei] for ei in idxs)
        sp = dist[t] * D
        if cur > 0 and (cur - sp) / cur < EPS:
            break
        d = {ei: y[ei] - x[ei] for ei in idxs}

        def phip(al):
            return sum(d[ei] * lat_of(ei, x[ei] + al * d[ei]) for ei in idxs)

        if phip(1.0) <= 0:
            al = 1.0
        elif phip(0.0) >= 0:
            al = 0.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if phip(mid) > 0:
                    hi = mid
                else:
                    lo = mid
            al = 0.5 * (lo + hi)
        for ei in idxs:
            x[ei] = x[ei] + al * d[ei]
    return sum(a[ei] * x[ei] + b[ei] * (x[ei] ** (k + 1)) for ei in idxs)


def reachable(N, edges, s, t, kept):
    adj = [[] for _ in range(N)]
    for ei in kept:
        adj[edges[ei][0]].append(edges[ei][1])
    seen = [False] * N; stk = [s]; seen[s] = True
    while stk:
        u = stk.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True; stk.append(v)
    return seen[t]


def main():
    raw = sys.stdin.read().split()
    it = iter(raw)
    N = int(next(it)); M = int(next(it))
    s = int(next(it)); t = int(next(it)); k = int(next(it)); D = float(next(it))
    edges = []
    for _ in range(M):
        u = int(next(it)); v = int(next(it)); a = float(next(it)); b = float(next(it))
        edges.append((u, v, a, b))
    a = [e[2] for e in edges]
    b = [e[3] for e in edges]

    kept = set(range(M))
    best = equilibrium(N, edges, a, b, s, t, D, k, kept)
    for _ in range(4):
        improved = False
        # try the cheapest-benefit candidates first: sort by current-order is fine
        for ei in sorted(kept):
            trial = kept - {ei}
            if not reachable(N, edges, s, t, trial):
                continue
            c = equilibrium(N, edges, a, b, s, t, D, k, trial)
            if c is not None and c < best - 1e-6 * abs(best):
                best = c; kept = trial; improved = True
        if not improved:
            break

    print(" ".join(str(i + 1) for i in sorted(kept)))


main()
