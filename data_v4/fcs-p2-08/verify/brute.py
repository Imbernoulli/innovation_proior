import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    adj = [[] for _ in range(n + 1)]
    indeg = [0] * (n + 1)
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        adj[u].append(v)
        indeg[v] += 1

    # Sources = vertices with in-degree 0.
    sources = [u for u in range(1, n + 1) if indeg[u] == 0]

    # Exhaustively enumerate every path that starts at a source and report the
    # maximum number of edges on such a path. The graph is a DAG, so every walk
    # is a simple path and the recursion terminates. This is the literal
    # definition (no DP, no greedy), used only as an independent oracle.
    best = 0

    def dfs(u, depth):
        nonlocal best
        if depth > best:
            best = depth
        for v in adj[u]:
            dfs(v, depth + 1)

    for s in sources:
        dfs(s, 0)

    print(best)

if __name__ == "__main__":
    sys.setrecursionlimit(1000000)
    main()
