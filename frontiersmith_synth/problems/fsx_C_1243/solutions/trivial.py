# TIER: trivial
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

    # identity mapping, always detour every routed pair through one fixed
    # hub qubit, never bother checking for gate cancellation.
    def graph_center():
        best_v, best_ecc = 0, None
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            dq = deque([s])
            while dq:
                u = dq.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        dq.append(v)
            ecc = max(dist)
            if best_ecc is None or ecc < best_ecc:
                best_ecc = ecc
                best_v = s
        return best_v

    initial_mapping = list(range(n))
    pos = list(range(n))
    at = list(range(n))
    ops = []
    anchor = graph_center()

    def do_swap(u, v):
        lu, lv = at[u], at[v]
        at[u], at[v] = lv, lu
        pos[lu], pos[lv] = v, u
        ops.append("S %d %d" % (u, v))

    for i in range(1, m + 1):
        t, a, b = gates[i - 1]
        p = pos[a]
        if p != anchor:
            path1 = bfs_path(n, adj, p, anchor)
            for k in range(len(path1) - 1):
                do_swap(path1[k], path1[k + 1])
        q = pos[b]
        if q not in adj[anchor]:
            path2 = bfs_path(n, adj, anchor, q)
            for k in range(len(path2) - 2):
                do_swap(path2[k], path2[k + 1])
        ops.append("G %d" % i)

    out = [str(n), " ".join(map(str, initial_mapping)), str(len(ops))]
    out.extend(ops)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
