import sys

def main():
    data = sys.stdin.buffer.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    q = int(data[idx]); idx += 1
    val = [0] * (n + 1)
    for i in range(1, n + 1):
        val[i] = int(data[idx]); idx += 1
    adj = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        a = int(data[idx]); idx += 1
        b = int(data[idx]); idx += 1
        adj[a].append(b)
        adj[b].append(a)

    # parent / depth via BFS from node 1
    parent = [0] * (n + 1)
    depth = [0] * (n + 1)
    order = []
    visited = [False] * (n + 1)
    from collections import deque
    dq = deque([1])
    visited[1] = True
    parent[1] = 0
    while dq:
        u = dq.popleft()
        order.append(u)
        for w in adj[u]:
            if not visited[w]:
                visited[w] = True
                parent[w] = u
                depth[w] = depth[u] + 1
                dq.append(w)

    def path_nodes(u, v):
        # collect nodes on path u..v by walking up to LCA
        su, sv = set(), set()
        au, av = u, v
        pu, pv = [], []
        # bring both to same depth
        while depth[au] > depth[av]:
            pu.append(au); au = parent[au]
        while depth[av] > depth[au]:
            pv.append(av); av = parent[av]
        while au != av:
            pu.append(au); au = parent[au]
            pv.append(av); av = parent[av]
        lca = au
        res = pu + [lca] + list(reversed(pv))
        return res

    out = []
    for _ in range(q):
        t = int(data[idx]); idx += 1
        if t == 1:
            u = int(data[idx]); idx += 1
            v = int(data[idx]); idx += 1
            x = int(data[idx]); idx += 1
            for w in path_nodes(u, v):
                val[w] ^= x
        else:
            u = int(data[idx]); idx += 1
            v = int(data[idx]); idx += 1
            s = 0
            for w in path_nodes(u, v):
                s += val[w]
            out.append(str(s))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

main()
