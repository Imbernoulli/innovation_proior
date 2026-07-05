# TIER: trivial
# Identity placement + greedy shortest-path routing: for each gate, move logical
# a one step at a time toward logical b (smallest-index next hop) until adjacent.
# This reproduces the checker's internal baseline -> scores ~0.1.
import sys


def bfs_dist(adj, P, src):
    dist = [-1] * P
    dist[src] = 0
    q = [src]; h = 0
    while h < len(q):
        u = q[h]; h += 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    P = int(next(it)); E = int(next(it)); M = int(next(it)); L = int(next(it))
    adj_set = [set() for _ in range(P)]
    for _ in range(E):
        u = int(next(it)); v = int(next(it))
        adj_set[u].add(v); adj_set[v].add(u)
    adj = [sorted(s) for s in adj_set]
    gates = [(int(next(it)), int(next(it))) for _ in range(M)]

    dist = [bfs_dist(adj, P, s) for s in range(P)]

    loc = list(range(L)) + [-1] * (P - L)
    occ = list(range(L)) + [-1] * (P - L)

    out = []
    # placement line: occ[phys]
    out.append(" ".join(str(occ[p]) for p in range(P)))

    for (a, b) in gates:
        moves = []
        pa = loc[a]; pb = loc[b]
        while dist[pa][pb] > 1:
            nxt = None
            for n in adj[pa]:
                if dist[n][pb] == dist[pa][pb] - 1:
                    nxt = n
                    break
            la = occ[pa]; ln = occ[nxt]
            occ[pa] = ln; occ[nxt] = la
            if la != -1:
                loc[la] = nxt
            if ln != -1:
                loc[ln] = pa
            moves.append((pa, nxt))
            pa = loc[a]; pb = loc[b]
        parts = [str(len(moves))]
        for (p, q) in moves:
            parts.append("%d %d" % (p, q))
        out.append(" ".join(parts))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
