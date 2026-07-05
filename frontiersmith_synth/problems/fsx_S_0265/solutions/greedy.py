# TIER: greedy
# One-idea heuristic: repeatedly find the current s-t shortest path and delete the
# CHEAPEST link on it (if it keeps s-t connected). Cheap to compute, but blind to how
# much a deletion actually lengthens the route.
import sys, heapq

def read():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); k = int(next(it))
    s = int(next(it)); t = int(next(it))
    eu = [0]*m; ev = [0]*m; ew = [0]*m
    for e in range(m):
        eu[e] = int(next(it)); ev[e] = int(next(it)); ew[e] = int(next(it))
    return n, m, k, s, t, eu, ev, ew

def sp_path(n, adj, removed, s, t):
    INF = float("inf")
    dist = [INF]*n
    pe = [-1]*n  # predecessor edge id
    dist[s] = 0
    pq = [(0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for (v, w, eid) in adj[u]:
            if eid in removed:
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                pe[v] = eid
                heapq.heappush(pq, (nd, v))
    if dist[t] == INF:
        return None, INF
    # reconstruct edge ids on path
    path = []
    cur = t
    while cur != s:
        eid = pe[cur]
        path.append(eid)
        # step to other endpoint
        cur = eu_g[eid] if ev_g[eid] == cur else ev_g[eid]
    return path, dist[t]

def main():
    global eu_g, ev_g
    n, m, k, s, t, eu, ev, ew = read()
    eu_g, ev_g = eu, ev
    adj = [[] for _ in range(n)]
    for e in range(m):
        adj[eu[e]].append((ev[e], ew[e], e))
        adj[ev[e]].append((eu[e], ew[e], e))

    removed = set()
    for _ in range(k):
        path, d = sp_path(n, adj, removed, s, t)
        if path is None:
            break
        # candidate = cheapest edge on path whose removal keeps s-t connected
        best = None
        for eid in sorted(path, key=lambda e: ew[e]):
            removed.add(eid)
            _, dd = sp_path(n, adj, removed, s, t)
            removed.discard(eid)
            if dd != float("inf"):
                best = eid
                break
        if best is None:
            break
        removed.add(best)

    out = [str(len(removed))]
    out.extend(str(e) for e in removed)
    sys.stdout.write(" ".join(out) + "\n")

if __name__ == "__main__":
    main()
