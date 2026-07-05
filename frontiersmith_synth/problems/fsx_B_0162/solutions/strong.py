# TIER: strong
"""Nearest-first scheduling with a one-step lookahead: always execute the
currently-cheapest interaction, and when routing it, move whichever endpoint
leaves the smallest total remaining distance over all still-pending
interactions.  Reuses placement (no restore)."""
import sys
from collections import deque


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); M = int(next(it)); K = int(next(it))
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u = int(next(it)); v = int(next(it))
        adj[u].append(v); adj[v].append(u)
    req = [(int(next(it)), int(next(it))) for _ in range(K)]

    def bfs(s):
        d = [-1] * N
        d[s] = 0
        q = deque([s])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if d[y] < 0:
                    d[y] = d[x] + 1
                    q.append(y)
        return d

    distm = [bfs(s) for s in range(N)]

    def bfs_path(src, dst):
        prev = [-2] * N
        prev[src] = -1
        q = deque([src])
        while q:
            x = q.popleft()
            if x == dst:
                break
            for y in adj[x]:
                if prev[y] == -2:
                    prev[y] = x
                    q.append(y)
        path = []
        x = dst
        while x != -1:
            path.append(x)
            x = prev[x]
        path.reverse()
        return path

    occ = list(range(N)); loc = list(range(N))
    out = []
    pending = list(req)

    def simulate(mover, other, cur_loc, cur_occ):
        lloc = cur_loc[:]; locc = cur_occ[:]
        path = bfs_path(lloc[mover], lloc[other])
        sw = []
        for i in range(len(path) - 2):
            u, v = path[i], path[i + 1]
            la, lb = locc[u], locc[v]
            locc[u], locc[v] = lb, la
            lloc[la], lloc[lb] = v, u
            sw.append((u, v))
        return sw, lloc, locc

    def pending_cost(new_loc, skip_idx):
        c = 0
        for j, (x, y) in enumerate(pending):
            if j == skip_idx:
                continue
            c += distm[new_loc[x]][new_loc[y]]
        return c

    while pending:
        best_i = min(range(len(pending)),
                     key=lambda i: distm[loc[pending[i][0]]][loc[pending[i][1]]])
        a, b = pending[best_i]
        swA, locA, occA = simulate(a, b, loc, occ)
        swB, locB, occB = simulate(b, a, loc, occ)
        if pending_cost(locA, best_i) <= pending_cost(locB, best_i):
            sw, nloc, nocc = swA, locA, occA
        else:
            sw, nloc, nocc = swB, locB, occB
        for (u, v) in sw:
            out.append("S %d %d" % (u, v))
        loc[:] = nloc
        occ[:] = nocc
        out.append("G %d %d" % (a, b))
        pending.pop(best_i)

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
