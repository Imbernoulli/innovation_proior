# TIER: strong
"""Portfolio router. Simulates several no-undo strategies and emits the cheapest:
  - given order / distance-ascending / distance-descending static orders
  - a dynamic 'nearest-pair-first' order (recomputes current distances each step)
each with two move policies (move lot a toward b, or b toward a). Picks the schedule
with the fewest SWAP moves. Always <= the given-order greedy (which is one candidate)."""
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


def bfs_dist(adj, Q, src, dst):
    if src == dst:
        return 0
    seen = [False] * Q
    seen[src] = True
    dq = deque([(src, 0)])
    while dq:
        node, d = dq.popleft()
        for nb in adj[node]:
            if not seen[nb]:
                if nb == dst:
                    return d + 1
                seen[nb] = True
                dq.append((nb, d + 1))
    return 10 ** 9


def route_one(adj, Q, pos, occ, a, b, move_b, out):
    """Bring a,b adjacent (moving a toward b, or b toward a). Mutates pos/occ, appends ops."""
    if move_b:
        src, dst = pos[b], pos[a]
    else:
        src, dst = pos[a], pos[b]
    path = bfs_path(adj, Q, src, dst)
    cnt = 0
    for k in range(len(path) - 2):
        p, q = path[k], path[k + 1]
        out.append("S %d %d" % (p, q))
        la, lb = occ[p], occ[q]
        occ[p], occ[q] = lb, la
        pos[la], pos[lb] = q, p
        cnt += 1
    out.append("G %d %d" % (a, b))
    return cnt


def run_static(adj, Q, inter, order, move_b):
    pos = list(range(Q)); occ = list(range(Q))
    out = []; swaps = 0
    for idx in order:
        a, b = inter[idx]
        swaps += route_one(adj, Q, pos, occ, a, b, move_b, out)
    return swaps, out


def run_dynamic(adj, Q, inter, move_b):
    pos = list(range(Q)); occ = list(range(Q))
    out = []; swaps = 0
    remaining = list(range(len(inter)))
    while remaining:
        best = None; bestd = None
        for idx in remaining:
            a, b = inter[idx]
            d = bfs_dist(adj, Q, pos[a], pos[b])
            if bestd is None or d < bestd:
                bestd = d; best = idx
        remaining.remove(best)
        a, b = inter[best]
        swaps += route_one(adj, Q, pos, occ, a, b, move_b, out)
    return swaps, out


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

    given = list(range(K))
    d0 = [bfs_dist(adj, Q, inter[i][0], inter[i][1]) for i in range(K)]
    asc = sorted(given, key=lambda i: (d0[i], i))
    desc = sorted(given, key=lambda i: (-d0[i], i))

    best = None
    for move_b in (False, True):
        for order in (given, asc, desc):
            s, o = run_static(adj, Q, inter, order, move_b)
            if best is None or s < best[0]:
                best = (s, o)
        s, o = run_dynamic(adj, Q, inter, move_b)
        if s < best[0]:
            best = (s, o)

    out = best[1]
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
