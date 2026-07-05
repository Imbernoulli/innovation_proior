# TIER: greedy
# Interaction-aware initial placement (affinity-weighted onto central lattice
# region), then the SAME greedy shortest-path router as trivial.  Better
# placement keeps frequently-interacting qubits close -> fewer routing swaps.
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

    # affinity weights
    w = [[0] * L for _ in range(L)]
    deg = [0] * L
    for (a, b) in gates:
        w[a][b] += 1
        w[b][a] += 1
        deg[a] += 1
        deg[b] += 1

    # central physical qubit = min eccentricity
    ecc = [max(dist[s]) for s in range(P)]
    center = min(range(P), key=lambda s: (ecc[s], s))

    phys_of = [-1] * L      # logical -> physical
    occ = [-1] * P
    placed = []

    # seed with highest-degree logical at center
    start = max(range(L), key=lambda q: (deg[q], -q))
    phys_of[start] = center
    occ[center] = start
    placed.append(start)

    unplaced = set(range(L))
    unplaced.discard(start)

    while unplaced:
        # pick unplaced logical with max affinity to already-placed set
        best_q = None; best_aff = None
        for q in unplaced:
            aff = 0
            for r in placed:
                aff += w[q][r]
            key = (aff, deg[q], -q)
            if best_aff is None or key > best_aff:
                best_aff = key; best_q = q
        q = best_q
        # pick empty physical minimizing weighted distance to placed partners
        best_phys = None; best_cost = None
        for s in range(P):
            if occ[s] != -1:
                continue
            cost = 0
            for r in placed:
                if w[q][r]:
                    cost += w[q][r] * dist[s][phys_of[r]]
            key = (cost, s)
            if best_cost is None or key < best_cost:
                best_cost = key; best_phys = s
        phys_of[q] = best_phys
        occ[best_phys] = q
        placed.append(q)
        unplaced.discard(q)

    loc = phys_of[:] + [-1] * 0
    loc = list(phys_of)
    while len(loc) < L:
        loc.append(-1)

    out = [" ".join(str(occ[p]) for p in range(P))]

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
        for (p, qq) in moves:
            parts.append("%d %d" % (p, qq))
        out.append(" ".join(parts))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
