import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    s = int(data[idx]); idx += 1
    adj = [[] for _ in range(n + 1)]
    for _ in range(m):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        w = int(data[idx]); idx += 1
        adj[u].append((v, w))

    INF = float('inf')
    # best[v] = max over all SIMPLE paths s->v of (min edge weight along the path).
    # A simple path suffices: any cycle on a path only adds edges, which can only
    # lower (never raise) the running minimum, so dropping it is never worse.
    best = [None] * (n + 1)
    # source: empty path, bottleneck = +inf (min over empty set)
    best[s] = INF

    # DFS over simple paths, carrying the running minimum.
    visited = [False] * (n + 1)

    def dfs(u, cur_min):
        # cur_min is the bottleneck of the path taken to reach u (INF at the start).
        if best[u] is None or cur_min > best[u]:
            best[u] = cur_min
        visited[u] = True
        for (v, w) in adj[u]:
            if not visited[v]:
                dfs(v, min(cur_min, w))
        visited[u] = False

    dfs(s, INF)

    out = []
    for v in range(1, n + 1):
        if best[v] is None:
            out.append("UNREACHABLE")
        elif v == s and best[v] == INF:
            out.append("INF")
        else:
            out.append(str(best[v]))
    sys.stdout.write("\n".join(out) + "\n")

main()
