# TIER: strong
# Interaction-aware initial placement + LOOK-AHEAD router.  Each gate is made
# executable with the minimal (dist-1) swaps, but among the equal-cost ways of
# doing so (move a->b, move b->a, or meet-in-the-middle) we pick the final
# configuration that minimizes the summed distance of the next few upcoming
# gates.  Keeping future partners close reduces later swaps -> beats greedy.
import sys

WINDOW = 10


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


def read_input():
    data = sys.stdin.read().split()
    it = iter(data)
    P = int(next(it)); E = int(next(it)); M = int(next(it)); L = int(next(it))
    adj_set = [set() for _ in range(P)]
    for _ in range(E):
        u = int(next(it)); v = int(next(it))
        adj_set[u].add(v); adj_set[v].add(u)
    adj = [sorted(s) for s in adj_set]
    gates = [(int(next(it)), int(next(it))) for _ in range(M)]
    return P, E, M, L, adj_set, adj, gates


def place(P, L, adj, dist, gates):
    w = [[0] * L for _ in range(L)]
    deg = [0] * L
    for (a, b) in gates:
        w[a][b] += 1; w[b][a] += 1; deg[a] += 1; deg[b] += 1
    ecc = [max(dist[s]) for s in range(P)]
    center = min(range(P), key=lambda s: (ecc[s], s))
    phys_of = [-1] * L
    occ = [-1] * P
    placed = []
    start = max(range(L), key=lambda q: (deg[q], -q))
    phys_of[start] = center; occ[center] = start; placed.append(start)
    unplaced = set(range(L)); unplaced.discard(start)
    while unplaced:
        best_q = None; best_key = None
        for q in unplaced:
            aff = sum(w[q][r] for r in placed)
            key = (aff, deg[q], -q)
            if best_key is None or key > best_key:
                best_key = key; best_q = q
        q = best_q
        best_phys = None; best_cost = None
        for s in range(P):
            if occ[s] != -1:
                continue
            cost = sum(w[q][r] * dist[s][phys_of[r]] for r in placed if w[q][r])
            key = (cost, s)
            if best_cost is None or key < best_cost:
                best_cost = key; best_phys = s
        phys_of[q] = best_phys; occ[best_phys] = q; placed.append(q); unplaced.discard(q)
    loc = list(phys_of)
    while len(loc) < L:
        loc.append(-1)
    return occ, loc


def step_toward(adj, dist, occ, loc, mover, target_phys):
    """Move logical `mover` one hop along a shortest path toward target_phys.
    Mutates occ,loc; returns the (p,q) swap performed."""
    pa = loc[mover]
    nxt = None
    for n in adj[pa]:
        if dist[n][target_phys] == dist[pa][target_phys] - 1:
            nxt = n
            break
    la = occ[pa]; ln = occ[nxt]
    occ[pa] = ln; occ[nxt] = la
    if la != -1:
        loc[la] = nxt
    if ln != -1:
        loc[ln] = pa
    return (pa, nxt)


def route_candidate(adj, dist, occ, loc, a, b, mode):
    """Return (moves, occ2, loc2) for making a,b adjacent under a given mode,
    without mutating the inputs."""
    o = list(occ); l = list(loc)
    moves = []
    turn = 0
    while dist[l[a]][l[b]] > 1:
        if mode == 0:                      # move a toward b
            mv = step_toward(adj, dist, o, l, a, l[b])
        elif mode == 1:                    # move b toward a
            mv = step_toward(adj, dist, o, l, b, l[a])
        else:                              # meet in the middle
            if turn % 2 == 0:
                mv = step_toward(adj, dist, o, l, a, l[b])
            else:
                mv = step_toward(adj, dist, o, l, b, l[a])
            turn += 1
        moves.append(mv)
    return moves, o, l


def main():
    P, E, M, L, adj_set, adj, gates = read_input()
    dist = [bfs_dist(adj, P, s) for s in range(P)]
    occ, loc = place(P, L, adj, dist, gates)

    out = [" ".join(str(occ[p]) for p in range(P))]

    for gi, (a, b) in enumerate(gates):
        future = gates[gi + 1: gi + 1 + WINDOW]
        best = None
        for mode in (0, 1, 2):
            moves, o2, l2 = route_candidate(adj, dist, occ, loc, a, b, mode)
            fcost = 0
            for (x, y) in future:
                fcost += dist[l2[x]][l2[y]]
            key = (fcost, mode)
            if best is None or key < best[0]:
                best = (key, moves, o2, l2)
        _, moves, occ, loc = best
        parts = [str(len(moves))]
        for (p, qq) in moves:
            parts.append("%d %d" % (p, qq))
        out.append(" ".join(parts))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
