# TIER: strong
# Steepest-ascent interdiction: at each of the k rounds, look at every link on the
# current shortest path, tentatively delete it, and keep the deletion that MAXIMISES
# the resulting s-t shortest path (rejecting any that disconnect s and t). This weighs
# the detour cost a deletion forces, not merely the link's own latency -> beats greedy.
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

def sp(n, adj, removed, s, t, want_path):
    INF = float("inf")
    dist = [INF]*n
    pe = [-1]*n
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
    if not want_path:
        return [], dist[t]
    path = []
    cur = t
    while cur != s:
        eid = pe[cur]
        path.append(eid)
        cur = EU[eid] if EV[eid] == cur else EV[eid]
    return path, dist[t]

def main():
    global EU, EV
    n, m, k, s, t, eu, ev, ew = read()
    EU, EV = eu, ev
    adj = [[] for _ in range(n)]
    for e in range(m):
        adj[eu[e]].append((ev[e], ew[e], e))
        adj[ev[e]].append((eu[e], ew[e], e))

    removed = set()
    for _ in range(k):
        path, d = sp(n, adj, removed, s, t, True)
        if path is None:
            break
        best_eid = None
        best_val = d
        for eid in path:
            removed.add(eid)
            _, dd = sp(n, adj, removed, s, t, False)
            removed.discard(eid)
            if dd == float("inf"):
                continue
            if dd > best_val:
                best_val = dd
                best_eid = eid
        if best_eid is None:
            # no single deletion strictly helps; fall back to any connectivity-safe edge
            for eid in path:
                removed.add(eid)
                _, dd = sp(n, adj, removed, s, t, False)
                removed.discard(eid)
                if dd != float("inf"):
                    best_eid = eid
                    break
            if best_eid is None:
                break
        removed.add(best_eid)

    out = [str(len(removed))]
    out.extend(str(e) for e in removed)
    sys.stdout.write(" ".join(out) + "\n")

if __name__ == "__main__":
    main()
