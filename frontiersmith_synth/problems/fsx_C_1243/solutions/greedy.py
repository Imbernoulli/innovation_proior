# TIER: greedy
import sys
from collections import deque


def bfs_path(n, adj, src, dst):
    if src == dst:
        return [src]
    parent = [-2] * n
    parent[src] = -1
    dq = deque([src])
    while dq:
        u = dq.popleft()
        if u == dst:
            break
        for v in adj[u]:
            if parent[v] == -2:
                parent[v] = u
                dq.append(v)
    path = [dst]
    cur = dst
    while parent[cur] != -1:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def main():
    data = sys.stdin.read().split()
    it = iter(data)

    def nxt():
        return int(next(it))

    n = nxt(); m = nxt(); e = nxt()
    adj = [set() for _ in range(n)]
    for _ in range(e):
        u = nxt(); v = nxt()
        adj[u].add(v); adj[v].add(u)
    gates = []
    for _ in range(m):
        t = nxt(); a = nxt(); b = nxt()
        gates.append((t, a, b))

    # The obvious first pass: keep the identity layout (logical i on physical
    # i) and, for every gate in program order, insert the shortest SWAP path
    # needed right now. Locally optimal per gate; never revisits the layout,
    # never looks for cancelling gate pairs.
    initial_mapping = list(range(n))
    pos = list(range(n))
    at = list(range(n))
    ops = []

    def do_swap(u, v):
        lu, lv = at[u], at[v]
        at[u], at[v] = lv, lu
        pos[lu], pos[lv] = v, u
        ops.append("S %d %d" % (u, v))

    for i in range(1, m + 1):
        t, a, b = gates[i - 1]
        p, q = pos[a], pos[b]
        if q not in adj[p]:
            path = bfs_path(n, adj, p, q)
            for k in range(len(path) - 2):
                do_swap(path[k], path[k + 1])
        ops.append("G %d" % i)

    out = [str(n), " ".join(map(str, initial_mapping)), str(len(ops))]
    out.extend(ops)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
