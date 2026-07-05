# TIER: greedy
# No reset: keep channels where routing leaves them (SABRE-naive).  For each splice,
# route the first channel toward its partner along a shortest path in the LIVE mapping.
# Avoiding the reset overhead alone beats the trivial baseline; the evolving placement
# occasionally leaves channels usefully close for later splices.
import sys
from collections import deque


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); e = int(next(it)); q = int(next(it))
    adj = [[] for _ in range(n)]
    for _ in range(e):
        u = int(next(it)); v = int(next(it))
        adj[u].append(v); adj[v].append(u)
    placement = [int(next(it)) for _ in range(n)]
    ops = [(int(next(it)), int(next(it))) for _ in range(q)]

    token_at = list(placement)
    site_at = [0] * n
    for s, t in enumerate(placement):
        site_at[t] = s

    def bfs_path(src, dst):
        prev = [-2] * n
        prev[src] = -1
        dq = deque([src])
        while dq:
            x = dq.popleft()
            if x == dst:
                break
            for y in adj[x]:
                if prev[y] == -2:
                    prev[y] = x
                    dq.append(y)
        path = []
        x = dst
        while x != -1:
            path.append(x)
            x = prev[x]
        path.reverse()
        return path

    def do_swap(u, v, out):
        tu = token_at[u]; tv = token_at[v]
        token_at[u] = tv; token_at[v] = tu
        site_at[tu] = v; site_at[tv] = u
        out.append("S %d %d" % (u, v))

    out = []
    for k, (a, b) in enumerate(ops, start=1):
        sa = site_at[a]; sb = site_at[b]
        path = bfs_path(sa, sb)
        for j in range(len(path) - 2):
            do_swap(path[j], path[j + 1], out)
        out.append("G %d" % k)

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
