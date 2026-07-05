# TIER: greedy
"""Exploit commutativity: repeatedly execute every already-adjacent maneuver for
free, then route the CLOSEST pending pair (min coupling-map distance). Reordering
the (commuting) ZZ maneuvers this way beats the fixed-order baseline."""
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
    edge_set = set()
    for _ in range(m):
        u = ni(); v = ni()
        adj[u].add(v); adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    init = [ni() for _ in range(n)]
    pairs = [(ni(), ni()) for _ in range(k)]

    pos = list(init)
    inv = [0] * n
    for p in range(n):
        inv[pos[p]] = p
    pending = set(range(k))
    out = []

    def adjacent(g):
        a, b = pairs[g]
        pa, pb = inv[a], inv[b]
        return (min(pa, pb), max(pa, pb)) in edge_set

    def do_swap(u, v):
        out.append("SWAP %d %d" % (u, v))
        lu = pos[u]; lv = pos[v]
        pos[u] = lv; pos[v] = lu
        inv[lv] = u; inv[lu] = v

    while pending:
        # flush all currently-adjacent maneuvers (free)
        progressed = True
        while progressed:
            progressed = False
            for g in sorted(pending):
                if adjacent(g):
                    out.append("GATE %d" % g)
                    pending.discard(g)
                    progressed = True
        if not pending:
            break
        # pick closest pending pair
        best = None; bestd = None
        for g in sorted(pending):
            a, b = pairs[g]
            d = len(shortest_path(adj, inv[a], inv[b])) - 1
            if bestd is None or d < bestd:
                bestd = d; best = g
        a, b = pairs[best]
        path = shortest_path(adj, inv[a], inv[b])
        for i in range(len(path) - 2):
            do_swap(path[i], path[i + 1])
        # best is now adjacent; loop flushes it

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
