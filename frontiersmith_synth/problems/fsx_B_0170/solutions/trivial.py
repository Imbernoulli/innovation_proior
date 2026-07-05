# TIER: trivial
# Naive per-waggle router: process the schedule in the given order; for each
# required interaction, walk one bee along the BFS shortest path until it is
# adjacent to its partner, then perform the waggle.  This reproduces the
# checker's internal baseline exactly (ratio ~ 0.1).
import sys
from collections import deque


def bfs_path(adj, src, dst, nq):
    if src == dst:
        return [src]
    par = [-2] * nq
    par[src] = -1
    q = deque([src])
    while q:
        x = q.popleft()
        for y in adj[x]:
            if par[y] == -2:
                par[y] = x
                if y == dst:
                    path = [dst]; c = dst
                    while par[c] != -1:
                        c = par[c]; path.append(c)
                    path.reverse(); return path
                q.append(y)
    return None


def main():
    data = sys.stdin.read().split()
    idx = 0
    nq = int(data[idx]); ne = int(data[idx + 1]); idx += 2
    adj = [[] for _ in range(nq)]
    for _ in range(ne):
        a = int(data[idx]); b = int(data[idx + 1]); idx += 2
        adj[a].append(b); adj[b].append(a)
    m = int(data[idx]); idx += 1
    req = []
    for _ in range(m):
        u = int(data[idx]); v = int(data[idx + 1]); idx += 2
        req.append((u, v))
    for a in range(nq):
        adj[a].sort()

    pos = list(range(nq))
    loc = list(range(nq))
    out = []
    for (u, v) in req:
        path = bfs_path(adj, loc[u], loc[v], nq)
        for i in range(len(path) - 2):
            a = path[i]; b = path[i + 1]
            la = pos[a]; lb = pos[b]
            pos[a] = lb; pos[b] = la
            loc[la] = b; loc[lb] = a
            out.append("S %d %d" % (a, b))
        out.append("G %d %d" % (path[-2], path[-1]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
