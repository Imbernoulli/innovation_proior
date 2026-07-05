# TIER: greedy
"""No-undo routing in the given order: after bringing lot a adjacent to b, keep the
new placement instead of restoring it, so later interactions inherit useful positions.
Beats the route-and-undo baseline by sharing state."""
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

    pos = list(range(Q))
    occ = list(range(Q))
    out = []
    for (a, b) in inter:
        path = bfs_path(adj, Q, pos[a], pos[b])
        for k in range(len(path) - 2):
            p, q = path[k], path[k + 1]
            out.append("S %d %d" % (p, q))
            la, lb = occ[p], occ[q]
            occ[p], occ[q] = lb, la
            pos[la], pos[lb] = q, p
        out.append("G %d %d" % (a, b))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
