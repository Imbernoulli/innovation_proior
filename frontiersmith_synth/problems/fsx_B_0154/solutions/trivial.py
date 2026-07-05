# TIER: trivial
"""Reproduce the checker's in-order greedy baseline exactly -> score ~0.1.
Process interactions in the given order; for each, move logical qubit a along the
shortest coupling-map path toward b (never undoing), then execute the maneuver."""
import sys
from collections import deque


def shortest_path(adj, s, t):
    if s == t:
        return [s]
    prev = {s: None}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in sorted(adj[u]):
            if v not in prev:
                prev[v] = u
                if v == t:
                    path = [t]
                    while path[-1] != s:
                        path.append(prev[path[-1]])
                    return path[::-1]
                q.append(v)
    return None


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    ni = lambda: int(next(it))
    n = ni(); m = ni(); k = ni()
    adj = [set() for _ in range(n)]
    for _ in range(m):
        u = ni(); v = ni()
        adj[u].add(v); adj[v].add(u)
    init = [ni() for _ in range(n)]
    pairs = [(ni(), ni()) for _ in range(k)]

    pos = list(init)
    inv = [0] * n
    for p in range(n):
        inv[pos[p]] = p
    out = []
    for g, (a, b) in enumerate(pairs):
        path = shortest_path(adj, inv[a], inv[b])
        for i in range(len(path) - 2):
            u = path[i]; v = path[i + 1]
            out.append("SWAP %d %d" % (u, v))
            lu = pos[u]; lv = pos[v]
            pos[u] = lv; pos[v] = lu
            inv[lv] = u; inv[lu] = v
        out.append("GATE %d" % g)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
