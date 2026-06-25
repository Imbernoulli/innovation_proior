import sys
from collections import deque

MOD = 1000000007

def main():
    sys.setrecursionlimit(100000)
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    nbr = [set() for _ in range(n + 1)]
    for _ in range(m):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        if u == v:
            continue
        nbr[u].add(v)
        nbr[v].add(u)

    # Plain BFS just to learn the shortest distance D from 1 to n.
    INF = float('inf')
    dist = [INF] * (n + 1)
    dist[1] = 0
    dq = deque([1])
    while dq:
        u = dq.popleft()
        for v in nbr[u]:
            if dist[v] == INF:
                dist[v] = dist[u] + 1
                dq.append(v)
    if dist[n] == INF:
        print(0)
        return
    D = dist[n]

    # Fully independent count: brute-force enumerate ALL simple paths from 1 to n
    # of length exactly D (a shortest path is simple), via DFS with a visited mask.
    # For the small generator sizes this terminates quickly. No layering tricks,
    # no per-node accumulation -- it literally walks every candidate route.
    visited = [False] * (n + 1)
    total = 0

    def dfs(u, depth):
        nonlocal total
        if depth == D:
            if u == n:
                total += 1
            return
        if u == n:
            # reached n too early; a longer route to n cannot be shortest, but a
            # route that passes through n is allowed to continue only if n is not
            # the destination's role here. Since we count routes ENDING at n with
            # length exactly D, stopping early at n with depth<D is just not counted.
            pass
        for v in nbr[u]:
            if not visited[v]:
                visited[v] = True
                dfs(v, depth + 1)
                visited[v] = False

    visited[1] = True
    dfs(1, 0)
    print(total % MOD)

main()
