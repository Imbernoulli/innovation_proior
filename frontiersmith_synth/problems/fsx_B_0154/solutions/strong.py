# TIER: strong
"""Closest-pair routing with a one-step lookahead on the move DIRECTION, plus
opportunistic mid-route flushing.

Routing a pair to adjacency costs the same number of SWAPs whether we drag tug a
toward b or b toward a, but the two choices leave the constellation in different
states -- affecting how far apart every OTHER pending pair ends up. Strong
simulates both directions, keeps the one that minimizes the total remaining
coupling-map distance over all pending maneuvers, and flushes any maneuver that
becomes adjacent after each SWAP. This co-locates frequently-interacting tugs and
uses fewer total SWAPs than the plain closest-pair greedy."""
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


def dist(adj, s, t):
    return len(shortest_path(adj, s, t)) - 1


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

    def adjacent(inv_, g):
        a, b = pairs[g]
        pa, pb = inv_[a], inv_[b]
        return (min(pa, pb), max(pa, pb)) in edge_set

    def flush():
        progressed = True
        while progressed:
            progressed = False
            for g in sorted(pending):
                if adjacent(inv, g):
                    out.append("GATE %d" % g)
                    pending.discard(g)
                    progressed = True

    def apply_swap(pos_, inv_, u, v, log):
        if log is not None:
            log.append((u, v))
        lu = pos_[u]; lv = pos_[v]
        pos_[u] = lv; pos_[v] = lu
        inv_[lv] = u; inv_[lu] = v

    def route_moves(pos_, inv_, mover_logical, other_logical, log):
        # drag mover_logical along shortest path toward other_logical's slot
        while True:
            sp = inv_[mover_logical]; tp = inv_[other_logical]
            if (min(sp, tp), max(sp, tp)) in edge_set:
                break
            path = shortest_path(adj, sp, tp)
            apply_swap(pos_, inv_, path[0], path[1], log)

    def total_remaining(inv_):
        s = 0
        for g in pending:
            a, b = pairs[g]
            s += dist(adj, inv_[a], inv_[b])
        return s

    while pending:
        flush()
        if not pending:
            break
        best = None; bestd = None
        for g in sorted(pending):
            a, b = pairs[g]
            d = dist(adj, inv[a], inv[b])
            if bestd is None or d < bestd:
                bestd = d; best = g
        a, b = pairs[best]

        # lookahead: try moving a vs moving b; keep lower total remaining distance
        options = []
        for mover, other in ((a, b), (b, a)):
            tp = list(pos); ti = list(inv); tl = []
            route_moves(tp, ti, mover, other, tl)
            # evaluate remaining distance excluding 'best' (about to be executed)
            s = 0
            for g in pending:
                if g == best:
                    continue
                x, y = pairs[g]
                s += dist(adj, ti[x], ti[y])
            options.append((s, len(tl), (mover, other)))
        options.sort(key=lambda o: (o[0], o[1]))
        mover, other = options[0][2]

        # execute chosen direction one SWAP at a time, flushing opportunistically
        while best in pending and not adjacent(inv, best):
            sp = inv[mover]; tp = inv[other]
            path = shortest_path(adj, sp, tp)
            apply_swap(pos, inv, path[0], path[1], None)
            out.append("SWAP %d %d" % (path[0], path[1]))
            flush()

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
