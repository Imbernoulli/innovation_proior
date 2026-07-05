import sys, heapq

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def dijkstra(n, adj, s, t):
    INF = float("inf")
    dist = [INF] * n
    dist[s] = 0
    pq = [(0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == t:
            return d
        for (v, w) in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist[t]

def main():
    # ---- read instance ----
    with open(sys.argv[1]) as f:
        inp = f.read().split()
    it = iter(inp)
    try:
        n = int(next(it)); m = int(next(it)); k = int(next(it))
        s = int(next(it)); t = int(next(it))
        eu = [0] * m; ev = [0] * m; ew = [0] * m
        for e in range(m):
            eu[e] = int(next(it)); ev[e] = int(next(it)); ew[e] = int(next(it))
    except Exception:
        fail("bad input")

    # full-graph adjacency + internal baseline B = original s-t shortest path
    full = [[] for _ in range(n)]
    for e in range(m):
        full[eu[e]].append((ev[e], ew[e]))
        full[ev[e]].append((eu[e], ew[e]))
    B = dijkstra(n, full, s, t)
    if B == float("inf") or B <= 0:
        fail("degenerate instance")
    B = int(B)

    # ---- read participant output ----
    with open(sys.argv[2]) as f:
        out = f.read().split()
    if not out:
        fail("empty output")
    # strict integer parsing (rejects nan/inf/floats -> score 0)
    try:
        r = int(out[0])
    except Exception:
        fail("bad count")
    if str(r) != out[0].lstrip("+"):
        # reject things like "1.0"; require a clean integer token
        try:
            if float(out[0]) != r:
                fail("non-integer count")
        except Exception:
            fail("bad count")
    if r < 0 or r > k:
        fail("budget: removed %d links, budget is %d" % (r, k))
    if len(out) < 1 + r:
        fail("expected %d edge ids" % r)

    removed = set()
    for j in range(1, 1 + r):
        tok = out[j]
        try:
            eid = int(tok)
        except Exception:
            fail("bad edge id token %r" % tok)
        if eid < 0 or eid >= m:
            fail("edge id %d out of range [0,%d)" % (eid, m))
        if eid in removed:
            fail("duplicate edge id %d" % eid)
        removed.add(eid)

    if len(out) != 1 + r:
        fail("trailing tokens")

    # ---- build residual graph (removed links deleted) ----
    adj = [[] for _ in range(n)]
    for e in range(m):
        if e in removed:
            continue
        adj[eu[e]].append((ev[e], ew[e]))
        adj[ev[e]].append((eu[e], ew[e]))

    F = dijkstra(n, adj, s, t)
    if F == float("inf"):
        fail("connectivity: s and t disconnected after removal")
    F = int(F)
    if F < B:
        # cannot happen (only edges removed), guard anyway
        F = B

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
