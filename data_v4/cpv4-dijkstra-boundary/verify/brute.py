import sys
sys.setrecursionlimit(100000)

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    c = [0] * (n + 1)
    for v in range(1, n + 1):
        c[v] = int(data[idx]); idx += 1
    adj = [[] for _ in range(n + 1)]
    for _ in range(m):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        w = int(data[idx]); idx += 1
        adj[u].append((v, w))

    # Independent brute force: exhaustively enumerate every SIMPLE path
    # (no repeated vertex) starting at node 1, tracking the running arrival
    # time, and record the minimum legal arrival time at node n.
    #
    # Legality: when we are present at a node v at time t, we require t < c[v]
    # (strict). Edge weights are positive, so an optimal route never revisits a
    # node and never benefits from waiting; therefore enumerating simple paths
    # and taking the min arrival at n is exactly the answer.

    best = [float('inf')]
    visited = [False] * (n + 1)

    def dfs(u, t):
        # We are AT node u at time t; t < c[u] already guaranteed by caller.
        if u == n:
            if t < best[0]:
                best[0] = t
            return
        visited[u] = True
        for (v, w) in adj[u]:
            if visited[v]:
                continue
            nt = t + w
            if nt < c[v]:                # strict boundary check on arrival
                dfs(v, nt)
        visited[u] = False

    # Start at node 1 at time 0; must be legal there.
    if 0 < c[1]:
        dfs(1, 0)

    print(best[0] if best[0] != float('inf') else -1)

main()
