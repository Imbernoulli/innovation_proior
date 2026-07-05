# TIER: strong
# Nearest-first router with DIRECTION LOOKAHEAD: when routing the closest pair,
# it may walk bee u toward v OR v toward u.  Both cost the same SWAPs now, but
# they leave DIFFERENT placements; strong simulates both and keeps the one that
# minimizes the total residual distance of all remaining interactions (a 1-step
# lookahead).  Better placement => cheaper future routing than plain greedy.
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


def apply_path(pos, loc, path):
    ops = []
    for i in range(len(path) - 2):
        a = path[i]; b = path[i + 1]
        la = pos[a]; lb = pos[b]
        pos[a] = lb; pos[b] = la
        loc[la] = b; loc[lb] = a
        ops.append((a, b))
    return ops


def residual(D, loc, rem, skip):
    tot = 0
    for j, (u, v) in enumerate(rem):
        if j == skip:
            continue
        tot += D[loc[u]][loc[v]]
    return tot


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

        # option A: walk u toward v
        posA = list(pos); locA = list(loc)
        pathA = bfs_path(adj, locA[u], locA[v], nq)
        opsA = apply_path(posA, locA, pathA)
        gA = (pathA[-2], pathA[-1])
        resA = residual(D, locA, rem, -1)

        # option B: walk v toward u
        posB = list(pos); locB = list(loc)
        pathB = bfs_path(adj, locB[v], locB[u], nq)
        opsB = apply_path(posB, locB, pathB)
        gB = (pathB[-2], pathB[-1])
        resB = residual(D, locB, rem, -1)

        if resB < resA:
            pos, loc = posB, locB
            ops, g = opsB, gB
        else:
            pos, loc = posA, locA
            ops, g = opsA, gA
        for (a, b) in ops:
            out.append("S %d %d" % (a, b))
        out.append("G %d %d" % (g[0], g[1]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
