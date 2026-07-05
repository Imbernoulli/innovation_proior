# TIER: trivial
"""Identity placement + per-task shortest-path routing (moves robot `a`).
Reproduces the checker's internal baseline exactly -> Ratio ~= 0.1."""
import sys
from collections import deque


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


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    V = int(next(it)); E = int(next(it)); m = int(next(it))
    adj = [set() for _ in range(V)]
    for _ in range(E):
        u = int(next(it)); v = int(next(it))
        adj[u].add(v); adj[v].add(u)
    tasks = [(int(next(it)), int(next(it))) for _ in range(m)]
    nbr = [sorted(a) for a in adj]

    pos = list(range(V))
    occ = list(range(V))
    steps = []
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
        steps.append("G")

    out = ["MAP " + " ".join(str(b) for b in range(V))]  # robot i on bay i
    out.extend(steps)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
