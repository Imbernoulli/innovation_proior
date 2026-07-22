import sys, heapq, math

# Deterministic checker for fsx_S_0513 (braess-pruning-equilibrium).
# Reads the network from <in>, the KEPT-edge subset from <out>, computes the
# Wardrop selfish-routing equilibrium on the kept subnetwork by Frank-Wolfe to a
# fixed iteration budget, and scores total travel time (minimize) against an
# internal single-path baseline.

CMAX = 250          # Frank-Wolfe iteration cap (equilibrium cost stable well before this)
EPS  = 1e-9         # relative-gap stopping tolerance


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def dijkstra(N, adj, src, lat):
    INF = float('inf')
    dist = [INF] * N
    pred = [-1] * N
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for (ei, v) in adj[u]:
            nd = d + lat[ei]
            if nd < dist[v] - 1e-15:
                dist[v] = nd
                pred[v] = ei
                heapq.heappush(pq, (nd, v))
    return dist, pred


def equilibrium(N, edges, s, t, D, k, kept):
    """Frank-Wolfe Wardrop equilibrium; returns total travel time or None if s-t
    unreachable within the kept subnetwork."""
    idxs = list(kept)
    adj = [[] for _ in range(N)]
    for ei in idxs:
        u, v, a, b = edges[ei]
        adj[u].append((ei, v))
    a = [e[2] for e in edges]
    b = [e[3] for e in edges]

    def lat_of(ei, x):
        return a[ei] + b[ei] * (x ** k)

    lat = [0.0] * len(edges)
    for ei in idxs:
        lat[ei] = lat_of(ei, 0.0)
    dist, pred = dijkstra(N, adj, s, lat)
    if dist[t] == float('inf'):
        return None

    x = {ei: 0.0 for ei in idxs}
    v = t                                    # all-or-nothing free-flow init
    while v != s:
        ei = pred[v]; x[ei] += D; v = edges[ei][0]

    for _ in range(CMAX):
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
        u, v, a, b = edges[ei]
        adj[u].append(v)
    seen = [False] * N
    st = [s]; seen[s] = True
    while st:
        u = st.pop()
        if u == t:
            return True
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True; st.append(v)
    return seen[t]


def main():
    try:
        raw = open(sys.argv[1]).read().split()
        it = iter(raw)
        N = int(next(it)); M = int(next(it))
        s = int(next(it)); t = int(next(it)); k = int(next(it))
        D = float(next(it))
        edges = []
        for _ in range(M):
            u = int(next(it)); v = int(next(it))
            a = float(next(it)); b = float(next(it))
            edges.append((u, v, a, b))
    except Exception:
        fail("bad input")

    # ---- parse participant output: 1-based kept edge indices ----
    otoks = open(sys.argv[2]).read().split()
    if len(otoks) > M + 1:                      # bounded read: never more than the whole edge set
        fail("too many tokens")
    kept = set()
    for tok in otoks:
        try:
            idx = int(tok)
        except ValueError:
            fail("non-integer token %r" % tok)   # rejects nan/inf/garbage
        if idx < 1 or idx > M:
            fail("edge index out of range %d" % idx)
        kept.add(idx - 1)

    if not kept or not reachable(N, edges, s, t, kept):
        fail("kept subnetwork does not route s->t")

    F = equilibrium(N, edges, s, t, D, k, kept)
    if F is None or not math.isfinite(F) or F <= 0:
        fail("infeasible / non-finite equilibrium")

    # ---- internal baseline B: route all demand on the free-flow shortest path ----
    adj = [[] for _ in range(N)]
    for ei, (u, v, a, b) in enumerate(edges):
        adj[u].append((ei, v))
    lat0 = [edges[i][2] for i in range(M)]
    dist, pred = dijkstra(N, adj, s, lat0)
    base_edges = set()
    vv = t
    while vv != s:
        ei = pred[vv]; base_edges.add(ei); vv = edges[ei][0]
    B = equilibrium(N, edges, s, t, D, k, base_edges)
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.3f B=%.3f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
