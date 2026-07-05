# TIER: strong
"""Evolving layout, no undo, and a nearest-first schedule: at each step pick the
still-pending hand-off whose two drones are CLOSEST on the current layout and
route it.  Executing cheap (already-nearby) hand-offs first keeps the layout from
drifting expensively and reuses positions -> markedly fewer swaps than input
order."""
import sys
from collections import deque


def parse():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); m = int(next(it)); k = int(next(it))
    adj = [[] for _ in range(n)]
    for _ in range(m):
        u = int(next(it)); v = int(next(it))
        adj[u].append(v); adj[v].append(u)
    req = [(int(next(it)), int(next(it))) for _ in range(k)]
    for u in range(n):
        adj[u].sort()
    return n, adj, req


def bfs_full(adj, s, n):
    dist = [-1] * n; par = [-2] * n
    dist[s] = 0; par[s] = -1
    dq = deque([s])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1; par[v] = u; dq.append(v)
    return dist, par


def main():
    n, adj, req = parse()
    pos = list(range(n))
    occ = list(range(n))
    pending = list(range(len(req)))
    out = []
    while pending:
        # pick pending hand-off with smallest current graph distance
        best = None; best_d = None; best_par = None
        for idx in pending:
            a, b = req[idx]
            dist, par = bfs_full(adj, pos[a], n)
            d = dist[pos[b]]
            if best_d is None or d < best_d or (d == best_d and idx < best):
                best, best_d, best_par = idx, d, par
        a, b = req[best]
        pa, pb = pos[a], pos[b]
        # reconstruct path pa->pb via best_par
        path = []; x = pb
        while x != -1:
            path.append(x); x = best_par[x]
        path.reverse()
        for i in range(len(path) - 2):
            x, y = path[i], path[i + 1]
            da, db = occ[x], occ[y]
            occ[x], occ[y] = db, da
            pos[da], pos[db] = y, x
            out.append("SWAP %d %d" % (x, y))
        out.append("APPLY %d %d" % (pos[a], pos[b]))
        pending.remove(best)
    out.append("END")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
