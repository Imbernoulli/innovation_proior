# TIER: trivial
"""Route each interaction from the identity placement, then RESTORE all tokens
before the next interaction.  Reproduces the checker's route-then-restore
baseline exactly -> ~0.1."""
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

    occ = list(range(N)); loc = list(range(N))
    out = []

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

    def do_swap(u, v):
        la, lb = occ[u], occ[v]
        occ[u], occ[v] = lb, la
        loc[la], loc[lb] = v, u
        out.append("S %d %d" % (u, v))

    for a, b in req:
        path = bfs_path(loc[a], loc[b])
        used = []
        for i in range(len(path) - 2):     # bring token a adjacent to b
            do_swap(path[i], path[i + 1])
            used.append((path[i], path[i + 1]))
        out.append("G %d %d" % (a, b))
        for (u, v) in reversed(used):       # restore to identity
            do_swap(u, v)

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
