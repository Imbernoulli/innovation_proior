# TIER: strong
"""Zone-aware placement + local-search refinement of the embedding + lookahead
routing (for each handoff move the endpoint with FEWER remaining future
handoffs), selecting the best of several candidate (placement, routing-policy)
strategies.  Distinct per-test behaviour from `greedy`, and fewer SWAPs."""
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


def local_search(V, w, D, pos, passes=4):
    pos = list(pos)

    def cost_if(l, bay):
        return sum(wt * D[bay][pos[nb]] for nb, wt in w[l].items())

    for _ in range(passes):
        improved = False
        for l1 in range(V):
            for l2 in range(l1 + 1, V):
                b1, b2 = pos[l1], pos[l2]
                before = cost_if(l1, b1) + cost_if(l2, b2)
                after = cost_if(l1, b2) + cost_if(l2, b1)
                if after < before - 1e-9:
                    pos[l1], pos[l2] = b2, b1
                    improved = True
        if not improved:
            break
    return pos


def simulate(V, nbr, tasks, placement, lookahead):
    pos = list(placement)
    occ = [0] * V
    for r in range(V):
        occ[pos[r]] = r
    rem = [0] * V
    if lookahead:
        for (a, b) in tasks:
            rem[a] += 1; rem[b] += 1
    steps = []
    swaps = 0
    for (a, b) in tasks:
        if lookahead:
            rem[a] -= 1; rem[b] -= 1
            mover, anchor = (b, a) if rem[b] < rem[a] else (a, b)
        else:
            mover, anchor = a, b
        path = bfs_path(nbr, pos[mover], pos[anchor])
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

    ident = list(range(V))
    emb = embed(V, w, deg, D)
    emb_ls = local_search(V, w, D, emb, passes=4) if V <= 42 else emb

    placements = [ident, emb, emb_ls]
    best = None
    for placement in placements:
        for la in (False, True):
            sw, steps = simulate(V, nbr, tasks, placement, la)
            if best is None or sw < best[0]:
                best = (sw, steps, placement)

    out = ["MAP " + " ".join(str(best[2][r]) for r in range(V))]
    out.extend(best[1])
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
