import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = int(next(it))
    d = [int(next(it)) for _ in range(n)]

    # Independent brute force: BFS over reachable amounts 0..A.
    # Each "step" adds one stamp. The first level at which we reach A is the
    # minimum number of stamps. This is an obviously-correct shortest-path
    # (unit edge weights) search, structurally different from the DP table fill.
    INF = float('inf')
    dist = [INF] * (A + 1)
    dist[0] = 0
    from collections import deque
    q = deque([0])
    while q:
        v = q.popleft()
        for x in d:
            nv = v + x
            if nv <= A and dist[nv] == INF:
                dist[nv] = dist[v] + 1
                q.append(nv)
    print(-1 if dist[A] == INF else dist[A])

main()
