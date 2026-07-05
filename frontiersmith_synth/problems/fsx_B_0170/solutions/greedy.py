# TIER: greedy
# Nearest-first router: repeatedly pick the remaining interaction whose two bees
# are currently CLOSEST (static physical distance table), route one bee to its
# partner, waggle.  Because just-interacted bees end up adjacent, further copies
# of that pair become distance-1 and are consumed next for free -> automatic
# batching of multiplicities.  Beats the naive order.
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


def all_pairs(adj, nq):
    D = []
    for s in range(nq):
        dist = [-1] * nq
        dist[s] = 0
        q = deque([s])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if dist[y] < 0:
                    dist[y] = dist[x] + 1
                    q.append(y)
        D.append(dist)
    return D


def main():
    data = sys.stdin.read().split()
    idx = 0
    nq = int(data[idx]); ne = int(data[idx + 1]); idx += 2
    adj = [[] for _ in range(nq)]
    for _ in range(ne):
        a = int(data[idx]); b = int(data[idx + 1]); idx += 2
        adj[a].append(b); adj[b].append(a)
    m = int(data[idx]); idx += 1
    rem = []
    for _ in range(m):
        u = int(data[idx]); v = int(data[idx + 1]); idx += 2
        rem.append((u, v))
    for a in range(nq):
        adj[a].sort()

    D = all_pairs(adj, nq)
    pos = list(range(nq))
    loc = list(range(nq))
    out = []
    while rem:
        best = 0
        bd = D[loc[rem[0][0]]][loc[rem[0][1]]]
        for i in range(1, len(rem)):
            u, v = rem[i]
            d = D[loc[u]][loc[v]]
            if d < bd:
                bd = d; best = i
        u, v = rem.pop(best)
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
