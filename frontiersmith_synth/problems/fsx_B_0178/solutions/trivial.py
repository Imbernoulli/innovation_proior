# TIER: trivial
"""Reproduce the checker baseline: route each required hand-off independently on
a fresh (identity) layout via a shortest path, apply, then UNDO the swaps so the
swarm returns to its parked layout before the next hand-off.  F == baseline B."""
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
    out = []
    for a, b in req:
        # identity layout: drone a on hub a, drone b on hub b
        path = bfs_path(adj, a, b, n)
        if len(path) < 3:
            continue
        fwd = [(path[i], path[i + 1]) for i in range(len(path) - 2)]
        for (x, y) in fwd:
            out.append("SWAP %d %d" % (x, y))
        out.append("APPLY %d %d" % (path[-2], path[-1]))
        for (x, y) in reversed(fwd):
            out.append("SWAP %d %d" % (x, y))
    out.append("END")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
