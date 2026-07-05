# TIER: trivial
"""Route-and-undo, in the given order -- reproduces the checker's internal baseline.
For each required interaction, bring lot a along a shortest slot-path to be adjacent
to lot b, apply the interaction, then undo every SWAP (restoring the placement).
Scores ~0.1 by construction."""
import sys
from collections import deque


def bfs_path(adj, Q, src, dst):
    if src == dst:
        return [src]
    prev = [-1] * Q
    seen = [False] * Q
    seen[src] = True
    dq = deque([src])
    while dq:
        node = dq.popleft()
        for nb in adj[node]:
            if not seen[nb]:
                seen[nb] = True
                prev[nb] = node
                if nb == dst:
                    path = [dst]
                    while path[-1] != src:
                        path.append(prev[path[-1]])
                    path.reverse()
                    return path
                dq.append(nb)
    return [src]


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    nxt = lambda: int(next(it))
    Q = nxt(); E = nxt(); K = nxt()
    adj = [[] for _ in range(Q)]
    for _ in range(E):
        u = nxt(); v = nxt()
        adj[u].append(v); adj[v].append(u)
    for a in range(Q):
        adj[a].sort()
    inter = [(nxt(), nxt()) for _ in range(K)]

    pos = list(range(Q))  # identity, reset each interaction
    out = []
    for (a, b) in inter:
        path = bfs_path(adj, Q, pos[a], pos[b])  # pos is identity here
        swaps = []
        for k in range(len(path) - 2):
            out.append("S %d %d" % (path[k], path[k + 1]))
            swaps.append((path[k], path[k + 1]))
        out.append("G %d %d" % (a, b))
        for (x, y) in reversed(swaps):
            out.append("S %d %d" % (x, y))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
