# TIER: greedy
"""Route each hand-off in input order on the ONE evolving layout and DON'T undo:
later hand-offs benefit from wherever the swarm already drifted.  ~half the swaps
of trivial (no undo pass)."""
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


def bfs_path(adj, s, t, n):
    par = [-2] * n; par[s] = -1
    dq = deque([s])
    while dq:
        u = dq.popleft()
        if u == t:
            break
        for v in adj[u]:
            if par[v] == -2:
                par[v] = u; dq.append(v)
    path = []; x = t
    while x != -1:
        path.append(x); x = par[x]
    path.reverse()
    return path


def main():
    n, adj, req = parse()
    pos = list(range(n))  # pos[drone] = hub
    occ = list(range(n))  # occ[hub]   = drone
    out = []
    for a, b in req:
        pa, pb = pos[a], pos[b]
        path = bfs_path(adj, pa, pb, n)
        for i in range(len(path) - 2):
            x, y = path[i], path[i + 1]
            da, db = occ[x], occ[y]
            occ[x], occ[y] = db, da
            pos[da], pos[db] = y, x
            out.append("SWAP %d %d" % (x, y))
        out.append("APPLY %d %d" % (pos[a], pos[b]))
    out.append("END")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
