# TIER: greedy
"""Zone-aware placement (greedy graph embedding of the handoff graph onto the
floor) with an identity fallback, + per-task shortest-path routing (moves robot
`a`).  Gathers each work zone so far fewer SWAPs are needed than the baseline."""
import sys
from collections import deque


def read_inst():
    tok = sys.stdin.read().split()
    it = iter(tok)
    V = int(next(it)); E = int(next(it)); m = int(next(it))
    adj = [set() for _ in range(V)]
    for _ in range(E):
        u = int(next(it)); v = int(next(it))
        adj[u].add(v); adj[v].add(u)
    tasks = [(int(next(it)), int(next(it))) for _ in range(m)]
    return V, adj, tasks


def all_pairs(adj, V):
    nbr = [sorted(a) for a in adj]
    D = []
    for s in range(V):
        dist = [-1] * V
        dist[s] = 0
        q = deque([s])
        while q:
            x = q.popleft()
            for y in nbr[x]:
                if dist[y] < 0:
                    dist[y] = dist[x] + 1
                    q.append(y)
        D.append(dist)
    return D, nbr


def bfs_path(nbr, src, dst):
    if src == dst:
        return [src]
    par = {src: src}
    q = deque([src])
    while q:
        x = q.popleft()
        for y in nbr[x]:
            if y not in par:
                par[y] = x
                if y == dst:
                    p = [dst]
                    while p[-1] != src:
                        p.append(par[p[-1]])
                    p.reverse()
                    return p
                q.append(y)
    return None


def embed(V, w, deg, D):
    cent = [sum(D[bay]) for bay in range(V)]
    placed = {}
    free = set(range(V))
    order = sorted(range(V), key=lambda x: (-deg[x], x))
    c0 = min(free, key=lambda bay: (cent[bay], bay))
    placed[order[0]] = c0; free.discard(c0)
    for lg in order[1:]:
        pn = [(nb, wt) for nb, wt in w[lg].items() if nb in placed]
        best = None; bestcost = None
        for bay in free:
            cost = (sum(wt * D[bay][placed[nb]] for nb, wt in pn)
                    if pn else cent[bay])
            if bestcost is None or cost < bestcost or (cost == bestcost and bay < best):
                bestcost = cost; best = bay
        placed[lg] = best; free.discard(best)
    return [placed[r] for r in range(V)]


def simulate(V, nbr, tasks, placement):
    """Route with move-`a` policy; return (swaps, steps)."""
    pos = list(placement)
    occ = [0] * V
    for r in range(V):
        occ[pos[r]] = r
    steps = []
    swaps = 0
    for (a, b) in tasks:
        path = bfs_path(nbr, pos[a], pos[b])
        cur = path[0]
        for k in range(1, len(path) - 1):
            nxt = path[k]
            steps.append("S %d %d" % (cur, nxt))
            ra, rn = occ[cur], occ[nxt]
            occ[cur], occ[nxt] = rn, ra
            pos[ra], pos[rn] = nxt, cur
            cur = nxt
            swaps += 1
        steps.append("G")
    return swaps, steps


def main():
    V, adj, tasks = read_inst()
    D, nbr = all_pairs(adj, V)
    w = [dict() for _ in range(V)]
    deg = [0] * V
    for (a, b) in tasks:
        w[a][b] = w[a].get(b, 0) + 1
        w[b][a] = w[b].get(a, 0) + 1
        deg[a] += 1; deg[b] += 1

    candidates = [list(range(V)), embed(V, w, deg, D)]
    best = None
    for placement in candidates:
        sw, steps = simulate(V, nbr, tasks, placement)
        if best is None or sw < best[0]:
            best = (sw, steps, placement)

    out = ["MAP " + " ".join(str(best[2][r]) for r in range(V))]
    out.extend(best[1])
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
